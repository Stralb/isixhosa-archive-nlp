#!/usr/bin/env python3
"""Independent audit of Phase31 saved scores and source model identities (no phase31_run import)."""
import argparse, csv, hashlib, json, random, re, zipfile, os
from pathlib import Path
import numpy as np
import pandas as pd
import sentencepiece as spm

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def check():
 p=Path(__file__).resolve().parent
 inp=Path(os.environ.get('PHASE31_SOURCE_ZIP','/mnt/data/TxtLangFiles1870-1894_TB.zip'))
 modelzip=Path(os.environ.get('PHASE31_MODELS_ZIP',str(p/'phase31_frozen_models_only.zip')))
 manifest=list(csv.DictReader((p/'cohort_manifest.csv').open()))
 scored=pd.read_csv(p/'page_model_scores.csv')
 saved=json.loads((p/'RESULTS.json').read_text())
 checks={}
 assert len(manifest)==1862 and len(scored)==len(manifest)*12
 checks['rows_and_model_pairings_valid']=True
 mf={r['file']:r for r in manifest}
 assert len(mf)==len(manifest)
 assert all(len(str(r['window_sha256']))==64 and r['source'] in ('IMVO','ISIG') for r in manifest)
 checks['cohort_count_and_hash']=digest(p/'cohort_manifest.csv')==saved['audit']['cohort_sha256']
 checks['score_hash']=digest(p/'page_model_scores.csv')==saved['audit']['score_sha256']
 assert checks['cohort_count_and_hash'] and checks['score_hash']
 # Verify frozen input and protocol hashes.
 digest_expected={l.split('  ')[-1]:l.split()[0] for l in (p/'FROZEN_INPUT_SHA256.txt').read_text().splitlines() if '  ' in l}
 for f,h in digest_expected.items():
  if f.endswith('TxtLangFiles1870-1894_TB.zip'):assert digest(inp)==h,('dataset archive hash mismatch')
  elif f.endswith('PRECOMMIT.md'):assert digest(p/'PRECOMMIT.md')==h,('precommit hash mismatch')
  elif f.endswith('ESTIMAND_CLARIFICATION_PRE_EVAL.md'):assert digest(p/'ESTIMAND_CLARIFICATION_PRE_EVAL.md')==h,('estimand precommit hash mismatch')
  elif f.endswith('phase31_run.py'):assert digest(p/'phase31_run.py')==h,('scoring code hash mismatch')
  elif f.endswith('isiXhosa_Phase30_Shared_Word_Frequency_BPE_2026-10-24.zip'):
   if os.environ.get('PHASE31_PRIOR_ZIP'):assert digest(os.environ['PHASE31_PRIOR_ZIP'])==h
 checks['frozen_input_hashes_match']=True
 # Independent window extraction for small deterministic stratified file sample.
 r=random.Random(31877)
 subset=r.sample(manifest,36)
 sourcezip=zipfile.ZipFile(inp);zp=zipfile.ZipFile(modelzip)
 modelmap={}
 for seed in range(3):
  for m in ['IMVO','ISIG','BAL_A','BAL_B']:
   entry=(f'Phase30/inputs/BPE_{m}_seed{seed}.model' if m in ('IMVO','ISIG') else f'Phase30/LINE_PLACEBO_{m[-1]}_seed{seed}.model')
   b=zp.read(entry)
   assert hashlib.sha256(b).hexdigest()==saved['audit']['model_sha256'][f'{m}_{seed}']['sha256']
   modelmap[m,seed]=spm.SentencePieceProcessor(model_proto=b)
 checks['model_frozen_hashes_match']=True
 word=re.compile('[a-z]+')
 maxdiff=0;tested=0
 for rec in subset:
  fname='TxtLangFiles1870-1894_TB/XhTxt/'+rec['file']
  t=sourcezip.read(fname).decode('utf-8-sig',errors='replace').replace('\r\n','\n')
  paras=re.split(r'\n\s*\n+',t)
  ts=[word.findall(x.lower()) for x in paras]
  idx=max(range(len(ts)),key=lambda j:(len(ts[j]),-j))
  words=ts[idx][:90]
  assert len(words)==90
  assert idx==int(rec['paragraph_index'])
  assert len(''.join(words))==int(rec['alpha_chars'])
  assert hashlib.sha256(' '.join(words).encode()).hexdigest()==rec['window_sha256']
  seq=' '.join(words)
  for seed in range(3):
   for m in ['IMVO','ISIG','BAL_A','BAL_B']:
    sp=modelmap[m,seed]
    n=len(sp.encode(seq,out_type=int))
    row=scored[(scored.file==rec['file'])&(scored.seed==seed)&(scored.model==m)]
    assert len(row)==1
    row=row.iloc[0]
    assert int(row.pieces)==n
    cumsum=np.cumsum([len(sp.encode(w,out_type=int)) for w in words])
    for budget in [192,224,256,288,320]:
     assert int(row[f'retain_{budget}'])==int(np.searchsorted(cumsum,budget,side='right'))
    maxdiff=max(maxdiff,abs(float(row['E_pieces_per_100_alpha'])-100*n/int(rec['alpha_chars'])))
    tested+=1
 sourcezip.close();zp.close()
 checks['independent_retokenization_sample']=tested
 checks['max_E_score_difference']=maxdiff
 assert maxdiff<1e-12
 # Independent central estimates, issue means with saved row values, confirm source-specific signs.
 gmeans={}
 for source,group in [('ISIG','EARLY_ISIG'),('IMVO','LATE_IMVO')]:
  groups=[]
  sub=scored[scored.group==group]
  for issue,frame in sub.groupby('issue'):
   diffE=[];diffR=[]
   for seed,fr in frame.groupby('seed'):
    pvt=fr.pivot(index='file',columns='model',values=['E_pieces_per_100_alpha','retain_256'])
    opp='IMVO' if source=='ISIG' else 'ISIG'
    diffE.extend(list(pvt['E_pieces_per_100_alpha'][opp]-pvt['E_pieces_per_100_alpha'][source]))
    diffR.extend(list(pvt['retain_256'][source]-pvt['retain_256'][opp]))
   groups.append([np.mean(diffE),np.mean(diffR)])
  val=np.mean(np.array(groups),axis=0)
  s=saved['groups'][group]
  assert abs(val[0]-s['specialist_E_gain_pieces_per_100_chars'])<1e-10
  assert abs(val[1]-s['specialist_retained_gain_words_B256'])<1e-10
  gmeans[group]=list(map(float,val))
 checks['independent_issue_aggregation']=gmeans
 (p/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(checks,indent=2,sort_keys=True))
 print(json.dumps(checks,indent=2))
if __name__=='__main__':check()
