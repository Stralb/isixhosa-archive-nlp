#!/usr/bin/env python3
"""Independent Phase 2 verifier: parses train/val source ZIP afresh and recomputes
all 13,760 validation encodings and bigram costs without evaluating 1888+.
"""
from __future__ import annotations
import argparse,csv,datetime,hashlib,json,math,re,zipfile
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import sentencepiece as spm

WORDS=re.compile('[a-z]+')
FILE=re.compile(r'IMVO_(\d{4})-(\d\d)-(\d\d)_p\d{3}\.txt$')
K=13;WINDOW=32
EXPECTED_PROTOCOL='bc443d883fa1c2b8eb09ced5f87d8f65b352d55512ef2c594b1b23e64d496bf4'
EXPECTED_ARCHIVE='375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5'

def sha(b):return hashlib.sha256(b).hexdigest()
def sha_file(p):return sha(Path(p).read_bytes())
def readcsv(p):
 with Path(p).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def verify(archive,root):
 root=Path(root);out=root/'Phase2'/'outputs'
 assert sha_file(archive)==EXPECTED_ARCHIVE
 assert sha_file(root/'Phase1'/'FROZEN_PROTOCOL.md')==EXPECTED_PROTOCOL
 # Independent file-based source reader; deliberately skip future-year payloads.
 d={};payload_reads=0
 with zipfile.ZipFile(archive) as z:
  for member in z.namelist():
   if not member.endswith('.txt') or '__MACOSX' in member:continue
   path=member.split('/')
   if len(path)<2 or path[-2] not in ('EnTxt','XhTxt'):continue
   name=path[-1];m=FILE.fullmatch(name)
   if not m:continue
   year,mon,day=map(int,m.groups());view='en' if path[-2]=='EnTxt' else 'xh'
   if year==1885 or year==1887:
    w=WORDS.findall(z.read(member).decode('utf-8-sig',errors='replace').lower());payload_reads+=1
    d.setdefault(name,{'year':year,'month':mon,'issue':f'IMVO_{year}-{mon:02d}-{day:02d}','views':{}})['views'][view]=w
 train={n:r for n,r in d.items() if r['year']==1885 or (r['year']==1887 and r['month']<=9)}
 val={n:r for n,r in d.items() if r['year']==1887 and r['month']>=10}
 assert len(train)==342 and len(val)==48
 assert not {x['issue'] for x in train.values()} & {x['issue'] for x in val.values()}
 assert len(d)==390 and payload_reads==780
 train_shingles={v:set() for v in ('en','xh')}
 for r in train.values():
  for v in ('en','xh'):
   w=r['views'][v]
   if len(w)>=16:
    train_shingles[v].update(tuple(w[i:i+K]) for i in range(len(w)-K+1))
 # Independent validation-window reconstruction: simple loop, not prefix summation.
 windows={};manifest=readcsv(out/'validation_windows_manifest.csv')
 for row in manifest:
  k=(row['filename'],row['view'],int(row['start']))
  r=val[row['filename']];w=r['views'][row['view']];start=k[2]
  assert len(w)>=32 and 0<=start<=len(w)-32
  seq=w[start:start+32];text=' '.join(seq)
  assert row['issue']==r['issue'] and int(row['chars'])==sum(map(len,seq))
  assert sha(text.encode())==row['text_sha256']
  assert all(tuple(seq[i:i+K]) not in train_shingles[row['view']] for i in range(32-K+1))
  assert k not in windows
  windows[k]=dict(text=text,chars=int(row['chars']),text_sha256=row['text_sha256'],issue=row['issue'])
 assert len(windows)==344
 # Confirm the exact same deterministic five evenly-spaced starts; avoid using run_phase2 implementation.
 for file,r in val.items():
  for v in ('en','xh'):
   w=r['views'][v]
   if len(w)<32:continue
   valid=[]
   for start in range(len(w)-31):
    if not any(tuple(w[j:j+K]) in train_shingles[v] for j in range(start,start+20)):
     valid.append(start)
   if len(valid)<=5:chosen=valid
   else:chosen=[valid[round(i*(len(valid)-1)/4)] for i in range(5)]
   observed=[int(x['start']) for x in manifest if x['filename']==file and x['view']==v]
   assert chosen==observed,(file,v,chosen,observed)
 # Training LM: source text fixed, not dependent on tokenizer sample arm.
 training_docs=[];train_manifest=readcsv(out/'lm_training_manifest.csv')
 for row in train_manifest:
  r=train[row['filename']];w=r['views'][row['view']]
  assert len(w)>=40
  x=' '.join(w[:128]);assert sha(x.encode())==row['text_sha256']
  assert sum(map(len,w[:128]))==int(row['chars'])
  training_docs.append(x)
 assert len(training_docs)==545
 audits=readcsv(out/'model_training_manifest.csv');assert len(audits)==40
 expected_labels={f'{family}_{arm}_{seed}' for family in ('bpe','unigram') for arm in ('PP','PW','WP','WW') for seed in (441,442,443,444,445)}
 assert {r['model_name'] for r in audits}==expected_labels
 assert all(499970<=int(r['actual_alpha_chars'])<=500000 for r in audits)
 assert all(int(r['validation_unknown_tokens'])==0 for r in audits)
 assert all(int(r['model_vocab_size'])==2000 for r in audits)
 assert all(r['sample_hash_sha256'] for r in audits)
 all_scores=readcsv(out/'validation_scores.csv');assert len(all_scores)==13760
 score_by_model=defaultdict(list)
 for r in all_scores:score_by_model[f'{r["family"]}_{r["arm"]}_{r["seed"]}'].append(r)
 assert set(score_by_model)==expected_labels
 count_pieces=0;max_bits_error=0.;max_pieces_error=0;verified_scores=0;token_counts=set()
 for audit in audits:
  label=audit['model_name'];path=out/'models'/f'{label}.model'
  assert sha_file(path)==audit['model_sha256']
  assert sha_file(out/'models'/f'{label}.vocab')==audit['vocab_sha256']
  sp=spm.SentencePieceProcessor(model_file=str(path));assert sp.vocab_size()==2000
  bg=Counter();unigram=Counter();prev_tot=Counter();N=0
  for text in training_docs:
   ids=sp.encode(text,out_type=int);prior=-1
   for nxt in ids:
    bg[(prior,nxt)]+=1;prev_tot[prior]+=1;unigram[nxt]+=1;N+=1;prior=nxt
  assert N==int(audit['LM_training_tokens']);token_counts.add(N)
  num=0;unks=0
  for row in score_by_model[label]:
   k=(row['filename'],row['view'],int(row['start']))
   valrow=windows[k];assert row['issue']==valrow['issue']
   assert row['text_sha256']==valrow['text_sha256'] and int(row['chars'])==valrow['chars']
   tokens=sp.encode(valrow['text'],out_type=int)
   assert len(tokens)==int(row['pieces'])
   unknown=sum(t==sp.unk_id() for t in tokens);assert unknown==int(row['unknown'])
   for alpha in (1.,10.,100.):
    cost=0.;prior=-1;dnom=N+.1*sp.vocab_size()
    for token in tokens:
     p=(bg[(prior,token)]+alpha*(unigram[token]+.1)/dnom)/(prev_tot[prior]+alpha)
     assert 0<p<=1
     cost-=math.log2(p);prior=token
    max_bits_error=max(max_bits_error,abs(cost-float(row[f'bits_{int(alpha)}'])))
    assert abs(cost-float(row[f'bits_{int(alpha)}']))<1e-8,(label,k,alpha)
   num+=1;unks+=unknown;count_pieces+=len(tokens)
  assert num==344 and unks==0
  assert sha_file(out/'model_rows'/f'{label}_validation.csv')==audit['validation_csv_sha256']
  verified_scores+=num
 # Independent alpha aggregation grouped by model, view; equal-weight across groups.
 sums=defaultdict(lambda:defaultdict(float))
 for row in all_scores:
  key=(row['family'],row['arm'],row['seed'],row['view'])
  x=sums[key];x['chars']+=int(row['chars'])
  for a in (1,10,100):x[f'bits_{a}']+=float(row[f'bits_{a}'])
 assert len(sums)==80
 selection={str(a):np.mean([q[f'bits_{a}']/q['chars'] for q in sums.values()]) for a in (1,10,100)}
 selection={k:float(v) for k,v in selection.items()}
 assert min(selection,key=selection.get)=='100'
 saved=json.loads((out/'PHASE2_SELECTION_FREEZE.json').read_text())
 assert saved['selected_alpha']==100 and saved['heldout_content_reads']==0
 for a in (1,10,100):
  y=next(x for x in saved['alpha_criteria'] if x['alpha']==a)
  assert abs(y['equal_view_family_arm_seed_mean_bpc']-selection[str(a)])<1e-12
 result={'PASS':True,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_zip_sha256':sha_file(archive),
    'frozen_phase1_protocol_sha256':EXPECTED_PROTOCOL,'train_pages':342,'validation_pages':48,'train_issues':86,'validation_issues':12,
    'validation_windows':344,'duplicate_free_window_reconstruction':True,'train_lm_documents':545,
    'trained_tokenizer_models':40,'model_fingerprints_verified':40,'all_model_vocab_2000':True,'alpha_choice':100,
    'validation_scores_recomputed':verified_scores,'max_recomputed_bpc_row_error':max_bits_error,
    'validation_unk_count':0,'validation_pieces_recomputed':count_pieces,'heldout_future_text_reads':0,
    'alpha_scores_independent':selection,'LM_train_token_totals_min':min(token_counts),'LM_train_token_totals_max':max(token_counts)}
 (out/'PHASE2_INDEPENDENT_VERIFICATION.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--root',required=True);a=p.parse_args()
 print(json.dumps(verify(a.archive,a.root),indent=2),flush=True)
