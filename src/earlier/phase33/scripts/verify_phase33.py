#!/usr/bin/env python3
"""Independent spot-check from archive + complete saved-table integrity checks."""
import argparse,re,zipfile,hashlib,json,random,math
from pathlib import Path
from collections import Counter
import pandas as pd, numpy as np, sentencepiece as spm

REGEX=re.compile('[a-z]+')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--zip',required=True);ap.add_argument('--root',required=True); a=ap.parse_args()
 root=Path(a.root);out=root/'outputs'
 d=pd.read_csv(out/'page_scores.csv');training=pd.read_csv(out/'training_audit.csv');perseed=pd.read_csv(out/'per_seed_summary.csv')
 assert len(d)==37395 and len(training)==15
 assert not d.duplicated(['filename','view','policy','seed']).any()
 assert d.groupby(['filename','view']).size().eq(15).all()
 assert all(d.unknown.eq(0))
 assert set(d.year.unique())==set(range(1888,1895))
 assert (training.training_alpha_chars.between(499970,500000)).all()
 assert training.groupby('policy').seed.nunique().eq(5).all()
 assert all(abs(q.training_en_alpha_chars/q.training_alpha_chars-q.training_en_alpha_share)<1e-12 for q in training.itertuples())
 assert not ((training.policy=='P') & (training.training_en_alpha_share>.30)).any()
 assert sum(d.filename.str.startswith('IMVO_')==False)==0
 hashes={}
 for r in training.itertuples():
  filename=out/'models'/f'bpe_{r.policy}_{r.seed}.model';assert filename.exists()
  hashes[filename.name]=hashlib.sha256(filename.read_bytes()).hexdigest()
  assert hashes[filename.name]==r.model_sha256
 # independently read from ZIP and reconstruct historical plain-text test strings
 byname={};
 random.seed(31123)
 sampled=d.groupby(['year','view','policy'],group_keys=False).sample(n=2,random_state=12345).drop_duplicates(['filename','view','policy','seed'])
 # an additional unrestricted deterministic check across entire saved table
 sampled=pd.concat([sampled,d.sample(n=110,random_state=31123)]).drop_duplicates(['filename','view','policy','seed'])
 fns=set(sampled.filename)
 with zipfile.ZipFile(a.zip) as z:
  for path in z.namelist():
   if '__MACOSX' in path or not path.endswith('.txt'):continue
   parts=path.split('/')
   if len(parts)<3 or parts[-2] not in ('EnTxt','XhTxt') or parts[-1] not in fns:continue
   key=(parts[-1], 'en' if parts[-2]=='EnTxt' else 'xh')
   wordseq=REGEX.findall(z.read(path).decode('utf-8-sig',errors='replace').lower())
   if len(wordseq)<40:continue
   wordseq=wordseq[:128]
   byname[key]=' '.join(wordseq)
 assert len(byname)>=80,len(byname)
 model_cache={};checks=0
 for r in sampled.itertuples():
  key=(r.filename,r.view)
  if key not in byname:continue
  txt=byname[key];count=sum(map(len,txt.split(' ')))
  assert count==r.chars and len(txt.split(' '))==r.words
  k=(r.policy,int(r.seed))
  if k not in model_cache:
   model_cache[k]=spm.SentencePieceProcessor(model_file=str(out/'models'/f'bpe_{r.policy}_{r.seed}.model'))
  tok=model_cache[k].encode(txt,out_type=int)
  assert len(tok)==r.pieces
  assert sum(t==model_cache[k].unk_id() for t in tok)==r.unknown
  checks+=1
 # exact independent aggregate recomputation
 groups=d.groupby(['cohort' if 'cohort' in d else 'year','view','policy','seed'])
 if 'cohort' not in d:
  d['cohort']=np.where(d.year==1888,'heldout_1888','later_1889_1894')
 total_checks=0
 for q in perseed.itertuples():
  t=d[(d.cohort==q.cohort)&(d.view==q.view)&(d.policy==q.policy)&(d.seed==q.seed)]
  assert abs(100*t.pieces.sum()/t.chars.sum()-q.pieces_per_100)<1e-10
  assert abs(t.lm_bits_10.sum()/t.chars.sum()-q.bpc_10)<1e-10
  total_checks+=1
 result={'pass':True,'page_rows':len(d),'issue_years':sorted(int(y) for y in d.year.unique()),
         'observed_page_view_units':d.groupby(['filename','view']).ngroups,
         'token_recomputations_from_source':checks,'independent_saved_aggregate_recomputations':total_checks,
         'model_hashes_checked':len(hashes),'all_unknown_counts_zero':True,
         'sampling_precommit_sha256':hashlib.sha256((root/'PRECOMMIT.md').read_bytes()).hexdigest(),
         'notes':'Re-encodes source strings and checks all saved aggregates. LM probability matrix independently verified in separate script.'}
 (root/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
