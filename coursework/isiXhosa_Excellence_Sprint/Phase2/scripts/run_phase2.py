#!/usr/bin/env python3
"""Excellence Sprint Phase 2. Strict train/validation-only factorial fitting.

Never reads 1888+ archive text. Does not compute held-out metrics.
Replays the Phase 1 frozen factorial sampler without changing its RNG.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,sys,time
from pathlib import Path
from collections import Counter
import numpy as np
import sentencepiece as spm

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Phase1'/'scripts'))
from audit_phase1 import parse,enumerate_units,K,WINDOW
from factorial_sampler import ARMS,build_pool,draw,distributions

SEEDS=(441,442,443,444,445)
FAMILIES=('bpe','unigram')
ALPHAS=(1.,10.,100.)
BUDGET=500_000
VOCAB=2_000


def sha(b):return hashlib.sha256(b).hexdigest()
def sha_file(p):return sha(Path(p).read_bytes())
def jwrite(p,obj):
 p=Path(p);p.write_text(json.dumps(obj,indent=2,sort_keys=True,default=str)+'\n',encoding='utf-8')
def cw(p,rows,cols):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)
def cr(p):
 with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def clean_validation(train,val):
 """Up to 5 deterministic evenly spaced 32-word nonoverlapping-with-train windows/view."""
 train_units,_=enumerate_units(train);val_units,_=enumerate_units(val)
 byv={v:set() for v in ('en','xh')}
 for u in train_units:
  w=u['words'];byv[u['view']].update(tuple(w[i:i+K]) for i in range(len(w)-K+1))
 rows=[];audit=[]
 for u in val_units:
  w=u['words']; n=len(w)
  if n<WINDOW:
   audit.append(dict(filename=u['file'],view=u['view'],words=n,candidate_starts=0,clean_starts=0,chosen=0))
   continue
  # Prefix-sum exact 13-gram contamination flags for each 32-token candidate.
  contamination=np.fromiter((tuple(w[i:i+K]) in byv[u['view']] for i in range(n-K+1)),dtype=np.uint8,count=n-K+1)
  pref=np.concatenate(([0],np.cumsum(contamination)))
  starts=np.flatnonzero(pref[WINDOW-K+1:] - pref[:-(WINDOW-K+1)] == 0)
  chosen=[]
  if len(starts)<=5:chosen=starts.tolist()
  else:
   chosen=starts[np.round(np.linspace(0,len(starts)-1,5)).astype(int)].tolist()
  for j,start in enumerate(chosen):
   x=' '.join(w[start:start+WINDOW]);chs=sum(map(len,w[start:start+WINDOW]));assert chs>0
   assert not any(tuple(w[k:k+K]) in byv[u['view']] for k in range(start,start+WINDOW-K+1))
   rows.append(dict(filename=u['file'],date=u['issue'].split('_')[-1],issue=u['issue'],view=u['view'],start=start,ordinal=j,
       text=x,chars=chs,text_sha256=sha(x.encode('utf-8'))))
  audit.append(dict(filename=u['file'],view=u['view'],words=n,candidate_starts=max(n-WINDOW+1,0),clean_starts=len(starts),chosen=len(chosen)))
 # Every result within 1887 Q4 and no issue in training
 assert rows and set(r['issue'] for r in rows).isdisjoint({u['issue'] for u in train_units})
 return rows,audit,{v:len(byv[v]) for v in byv}


def build_lm_docs(train):
 docs=[]
 for rec in train:
  for view in ('en','xh'):
   words=rec['views'][view]
   if len(words)<40:continue
   text=' '.join(words[:128]);chs=sum(map(len,words[:128]))
   docs.append(dict(filename=rec['file'],issue=rec['issue'],view=view,text=text,chars=chs,text_sha256=sha(text.encode())))
 return docs

class SmoothedBigram:
 def __init__(self,sp,train_docs):
  self.big=Counter();self.first=Counter();self.uni=Counter();self.ntokens=0;self.vocab=sp.vocab_size()
  for doc in train_docs:
   prev=-1
   for token in sp.encode(doc['text'],out_type=int):
    self.big[(prev,token)]+=1;self.first[prev]+=1;self.uni[token]+=1;self.ntokens+=1;prev=token
  assert self.ntokens>0
 def bits(self,ids,alpha):
  p=-1;cost=0.;den=self.ntokens+.1*self.vocab
  for t in ids:
   base=(self.uni[t]+.1)/den
   prob=(self.big[(p,t)]+alpha*base)/(self.first[p]+alpha)
   assert 0.<prob<=1.
   cost-=math.log2(prob);p=t
  return cost


def train_sp(lines,prefix,family):
 training=prefix.with_suffix('.training.tmp')
 training.write_text('\n'.join(lines)+'\n',encoding='utf-8')
 try:
  spm.SentencePieceTrainer.train(input=str(training),model_prefix=str(prefix),model_type=family,
       vocab_size=VOCAB,character_coverage=1.,hard_vocab_limit=False,
       normalization_rule_name='identity',bos_id=-1,eos_id=-1,pad_id=-1,unk_id=0,
       num_threads=1,max_sentence_length=4096,input_sentence_size=0,
       shuffle_input_sentence=False,split_by_whitespace=True,add_dummy_prefix=True,
       remove_extra_whitespaces=True,minloglevel=2)
 finally:training.unlink(missing_ok=True)
 sp=spm.SentencePieceProcessor(model_file=str(prefix)+'.model')
 assert sp.vocab_size()>1800,(prefix,sp.vocab_size())
 return sp


def model_validation(sp,docs,val,family,arm,seed):
 lm=SmoothedBigram(sp,docs)
 rows=[]
 for r in val:
  toks=sp.encode(r['text'],out_type=int)
  rows.append(dict(family=family,arm=arm,seed=seed,filename=r['filename'],issue=r['issue'],view=r['view'],
    start=r['start'],ordinal=r['ordinal'],chars=r['chars'],text_sha256=r['text_sha256'],pieces=len(toks),
    unknown=sum(t==sp.unk_id() for t in toks),
    bits_1=lm.bits(toks,1.),bits_10=lm.bits(toks,10.),bits_100=lm.bits(toks,100.)))
 return rows,lm.ntokens


def summarize_validation(rows):
 # Equal-weight two views across 4 arms, 5 seeds and both families. No arm-specific alpha selection.
 pairs={}
 for r in rows:
  key=(r['family'],r['arm'],int(r['seed']),r['view'])
  if key not in pairs:pairs[key]={'chars':0,'bits_1':0.,'bits_10':0.,'bits_100':0.,'pieces':0,'unknown':0,'windows':0,'issues':set()}
  q=pairs[key];q['chars']+=int(r['chars']);q['pieces']+=int(r['pieces']);q['unknown']+=int(r['unknown']);q['windows']+=1;q['issues'].add(r['issue'])
  for a in (1,10,100):q[f'bits_{a}']+=float(r[f'bits_{a}'])
 assert len(pairs)==2*4*5*2,len(pairs)
 sums=[]
 for (fam,arm,seed,v),q in sorted(pairs.items()):
  sums.append(dict(family=fam,arm=arm,seed=seed,view=v,chars=q['chars'],windows=q['windows'],issues=len(q['issues']),
    pieces_per_100=100*q['pieces']/q['chars'],unk_per_100=100*q['unknown']/q['chars'],
    bpc_1=q['bits_1']/q['chars'],bpc_10=q['bits_10']/q['chars'],bpc_100=q['bits_100']/q['chars']))
 alpha_scores=[]
 for a in (1,10,100):
  x=np.array([r[f'bpc_{a}'] for r in sums])
  alpha_scores.append(dict(alpha=a,equal_view_family_arm_seed_mean_bpc=float(x.mean()),model_view_pairs=len(x),
    en_mean_bpc=float(np.mean([r[f'bpc_{a}'] for r in sums if r['view']=='en'])),
    xh_mean_bpc=float(np.mean([r[f'bpc_{a}'] for r in sums if r['view']=='xh']))))
 # Differences below 1e-12 treated as ties, favor alpha=10.
 mn=min(r['equal_view_family_arm_seed_mean_bpc'] for r in alpha_scores)
 near=[r for r in alpha_scores if r['equal_view_family_arm_seed_mean_bpc']<=mn+1e-12]
 selected=sorted(near,key=lambda r:(r['alpha']!=10,r['alpha']))[0]['alpha']
 return sums,alpha_scores,selected


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--archive',required=True);ap.add_argument('--output',required=True);ap.add_argument('--stop-after',type=int,default=0);args=ap.parse_args()
 out=Path(args.output);(out/'models').mkdir(parents=True,exist_ok=True);(out/'model_rows').mkdir(parents=True,exist_ok=True)
 assert sha_file(ROOT/'Phase1'/'FROZEN_PROTOCOL.md')=='bc443d883fa1c2b8eb09ced5f87d8f65b352d55512ef2c594b1b23e64d496bf4','Phase1 protocol has changed'
 assert sha_file(args.archive)=='375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5','Archive hash changed'
 train,val,_,reads=parse(args.archive)
 assert reads['heldout_view_content_reads']==0
 pool=build_pool(train,False,False)
 val_rows,val_audit,train_shingle_types=clean_validation(train,val)
 lm_docs=build_lm_docs(train)
 assert sum(r['view']=='en' for r in val_rows)>=40 and sum(r['view']=='xh' for r in val_rows)>=40
 assert sum(r['view']=='en' for r in lm_docs)>0 and sum(r['view']=='xh' for r in lm_docs)>0
 assert set(r['issue'] for r in val_rows).isdisjoint({r['issue'] for r in lm_docs})
 cw(out/'validation_windows_manifest.csv', [{k:v for k,v in r.items() if k!='text'} for r in val_rows],
    ['filename','date','issue','view','start','ordinal','chars','text_sha256'])
 cw(out/'validation_eligibility.csv',val_audit,['filename','view','words','candidate_starts','clean_starts','chosen'])
 cw(out/'lm_training_manifest.csv',[{k:v for k,v in r.items() if k!='text'} for r in lm_docs],['filename','issue','view','chars','text_sha256'])
 jwrite(out/'source_only_manifest.json',dict(archive_sha256=sha_file(args.archive),protocol_sha256=sha_file(ROOT/'Phase1'/'FROZEN_PROTOCOL.md'),
  train_pages=len(train),train_issues=len({r['issue'] for r in train}),validation_pages=len(val),validation_issues=len({r['issue'] for r in val}),
  train_pageviews=len(pool),train_13grams_by_view=train_shingle_types,validation_windows=len(val_rows),
  validation_windows_en=sum(r['view']=='en' for r in val_rows),validation_windows_xh=sum(r['view']=='xh' for r in val_rows),
  lm_train_docs=len(lm_docs),heldout_content_reads=reads['heldout_view_content_reads']))
 completed=0;total=len(FAMILIES)*len(SEEDS)*len(ARMS)
 t0=time.time()
 for family in FAMILIES:
  for seed in SEEDS:
   for arm in ARMS:
    label=f'{family}_{arm}_{seed}';prefix=out/'models'/label;metapath=out/'model_rows'/f'{label}.json';valcsv=out/'model_rows'/f'{label}_validation.csv'
    if metapath.exists() and valcsv.exists() and Path(str(prefix)+'.model').exists():
     record=json.loads(metapath.read_text())
     assert sha_file(str(prefix)+'.model')==record['model_sha256']
     assert sha_file(valcsv)==record['validation_csv_sha256']
     # Sample hash also re-derived for portable integrity.
     lines,replay=draw(pool,arm,seed,BUDGET)
     assert replay['sample_hash_sha256']==record['sample_hash_sha256']
     completed+=1
     print('REUSE',label,completed,'of',total,'elapsed',round(time.time()-t0,1),flush=True)
     continue
    lines,audit=draw(pool,arm,seed,BUDGET)
    sp=train_sp(lines,prefix,family)
    val_scores,lm_tokens=model_validation(sp,lm_docs,val_rows,family,arm,seed)
    cols=['family','arm','seed','filename','issue','view','start','ordinal','chars','text_sha256','pieces','unknown','bits_1','bits_10','bits_100']
    cw(valcsv,val_scores,cols)
    record={**audit,'family':family,'model_name':label,'model_sha256':sha_file(str(prefix)+'.model'),
      'vocab_sha256':sha_file(str(prefix)+'.vocab'),'model_vocab_size':sp.vocab_size(),
      'validation_csv_sha256':sha_file(valcsv),'validation_windows':len(val_scores),
      'validation_unknown_tokens':sum(int(r['unknown']) for r in val_scores),'LM_training_tokens':lm_tokens,
      'LM_training_documents':len(lm_docs),'normalization':'identity','character_coverage':1.,
      'train_years':'1885;1887-Jan-Sep','validation_period':'1887-Oct-Dec'}
    jwrite(metapath,record)
    completed+=1
    print('MODEL',label,completed,'of',total,'vocab',sp.vocab_size(),'EN share',round(audit['en_alpha_share'],4),
      'unknown',record['validation_unknown_tokens'],'elapsed',round(time.time()-t0,1),flush=True)
    if args.stop_after and completed>=args.stop_after:break
   if args.stop_after and completed>=args.stop_after:break
  if args.stop_after and completed>=args.stop_after:break
 if completed==total:
  recs=[json.loads(p.read_text()) for p in sorted((out/'model_rows').glob('*.json'))]
  score_rows=[]
  for p in sorted((out/'model_rows').glob('*_validation.csv')):score_rows.extend(cr(p))
  assert len(recs)==total
  assert len(score_rows)==total*len(val_rows)
  cw(out/'model_training_manifest.csv',recs,list(recs[0].keys()))
  cw(out/'validation_scores.csv',score_rows,list(score_rows[0].keys()))
  sums,alpha_scores,chosen=summarize_validation(score_rows)
  cw(out/'validation_model_view_summary.csv',sums,list(sums[0].keys()))
  cw(out/'alpha_selection.csv',alpha_scores,list(alpha_scores[0].keys()))
  decision={'selected_alpha':chosen,'global_selection':'equal-weight mean BPC across English and isiXhosa views; BPE and Unigram; all 4 arms and 5 seeds',
    'selection_data':'1887 Oct-Dec duplicate-free 32-word windows only','candidate_grid':list(ALPHAS),
    'selected_before_1888_1894_factorial_evaluation':True,'phase1_protocol_sha256':sha_file(ROOT/'Phase1'/'FROZEN_PROTOCOL.md'),
    'archive_sha256':sha_file(args.archive),'models_count':len(recs),'validation_scored_rows':len(score_rows),
    'validation_windows_per_model':len(val_rows),'heldout_content_reads':0,'alpha_criteria':alpha_scores,
    'status':'PHASE2_VALIDATION_FROZEN'}
  jwrite(out/'PHASE2_SELECTION_FREEZE.json',decision)
  print('SELECTION',json.dumps(decision,indent=2),flush=True)
  print('DONE elapsed',round(time.time()-t0,1),flush=True)

if __name__=='__main__':main()
