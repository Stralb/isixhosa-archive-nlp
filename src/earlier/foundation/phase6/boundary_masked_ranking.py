#!/usr/bin/env python3
"""Retrospective score robustness, not a new held-out experiment."""
import collections, json, re, sys, zipfile
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/data/isiXhosa_publication_continuation')
sys.path.insert(0,str(ROOT))
from audit_overlap import load_pages,digest
from pilot_charlm import doc_rep,normalize_weight_multipliers,build_model,eval_novel_masks
from selection_reversal import compute_word_losses,by_issue,get_summary,bootstrap
OUT=ROOT/'phase6'
RX=re.compile(r'[^\W\d_]+',re.UNICODE)

def build_paragraph_data(original):
 wanted=set(x['name'] for x in original)
 pmap={}
 with zipfile.ZipFile('/mnt/data/TxtLangFiles1870-1894_TB.zip') as z:
  for path in z.namelist():
   fn=path.rsplit('/',1)[-1]
   if fn not in wanted or '/XhTxt/' not in path or '/._' in path:continue
   txt=z.read(path).decode('utf8','replace')
   blocks=[RX.findall(x.casefold()) for x in re.split(r'\n\s*\n',txt)]
   blocks=[x for x in blocks if len(x)]
   assert [t for block in blocks for t in block] == next(p['tokens'] for p in original if p['name']==fn),fn
   pmap[fn]=blocks
 return pmap

def masks(pages,pmap):
 train=[x for x in pages if x['split']=='train']
 tix=set()
 for p in train:
  for block in pmap[p['name']]:
   for j in range(max(0,len(block)-11)):
    tix.add(digest(block[j:j+12]))
 kept={}
 for p in pages:
  labels=[]
  for block in pmap[p['name']]:
   cov=bytearray(len(block))
   for j in range(max(0,len(block)-11)):
    if digest(block[j:j+12]) in tix:
     cov[j:j+12]=b'\x01'*12
   labels.extend([not v for v in cov])
  assert len(labels)==len(p['tokens'])
  kept[p['name']]=np.asarray(labels,dtype=bool)
 return kept

def main():
 pages=load_pages()
 import audit_overlap
 train=[p for p in pages if p['split']=='train']
 ix,_=audit_overlap.index_ngrams(train,12)
 docs=[doc_rep(p,ix) for p in pages]
 trd=[d for d in docs if d['split']=='train']
 ev=[d for d in docs if d['split']!='train']
 eval_novel_masks(ev,ix)
 alph=set(c for d in trd for c in d['text'])|{'?'}
 fac=normalize_weight_multipliers(trd)
 models={}
 for policy in ('BALANCED','HARD'):
  m=build_model(trd,policy,fac,alph)
  models[policy]=compute_word_losses(m,ev,alph)
  print('evaluated',policy,flush=True)
 page_before={}
 for name in ('validation','old_test'):
  which=[d for d in ev if d['split']==name]
  result=get_summary(by_issue(models,which))
  page_before[name]=result
  print('old_page_'+name,result['BALANCED']['macro_nonmatch']-result['HARD']['macro_nonmatch'],flush=True)
 pmap=build_paragraph_data(pages)
 newm=masks(pages,pmap)
 for pol in models:
  for d in ev:
   models[pol][d['name']]['keep']=newm[d['name']]
 output={'status':'RETROSPECTIVE_NOT_INDEPENDENT','models':'unchanged interpolated character ngram, 4-character context, gamma=20',
         'policies':['BALANCED','HARD'],'training_weights':'unchanged page-concat original',
         'new_eval_masks':'within-blankline-paragraph 12-word exact overlap','splits':{}}
 for split in ('validation','old_test'):
  which=[d for d in ev if d['split']==split]
  issue=by_issue(models,which)
  summ=get_summary(issue); boot=bootstrap(issue,bs=2500,seed=20261022)
  delta=summ['BALANCED']['macro_nonmatch']-summ['HARD']['macro_nonmatch']
  res={'old_mask':page_before[split], 'paragraph_mask':summ,
       'delta_BALANCED_minus_HARD_nonmatch':delta,
       'bootstrap':boot['contrasts_BALANCED_minus_HARD']['macro_nonmatch'],
       'by_source_delta':{s:summ['BALANCED']['domains'][s]['nonmatch']-summ['HARD']['domains'][s]['nonmatch'] for s in ('IMVO','ISIG')},
       'old_delta':page_before[split]['BALANCED']['macro_nonmatch']-page_before[split]['HARD']['macro_nonmatch']}
  output['splits'][split]=res
  print('new_paragraph_'+split, 'delta',delta,'CI',res['bootstrap']['ci_95'],'src_deltas',res['by_source_delta'],flush=True)
 (OUT/'masked_selection_results.json').write_text(json.dumps(output,indent=2))
 print('saved',flush=True)
if __name__=='__main__':main()
