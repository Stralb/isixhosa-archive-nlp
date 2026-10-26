#!/usr/bin/env python3
"""Phase29C: character-mass reweighting by historical surface-token length, fixed BPE models."""
from pathlib import Path
import os
import numpy as np, collections,csv,json,hashlib
import sentencepiece as spm
D=Path(__file__).parent; P=Path(os.environ.get('PHASE26_DIR',str(D/'inputs'/'phase26')))
W=[json.loads(x) for x in (P/'windows.jsonl').read_text().splitlines() if '"year": 1888' in x]
assert len(W)==207
BINS=['1-3','4-6','7-9','10-12','13+']

def binidx(t):
 l=sum(ch.isalpha() for ch in t)
 return (0 if l<=3 else 1 if l<=6 else 2 if l<=9 else 3 if l<=12 else 4)
models={(s,k):spm.SentencePieceProcessor(model_file=str(P/f'BPE_{s}_seed{k}.model')) for k in range(3) for s in ('IMVO','ISIG')}
cache={key:{} for key in models}

def count(key,t):
 ca=cache[key]
 if t not in ca:ca[t]=len(models[key].encode(t,out_type=int))
 return ca[t]

records=[]; add_errors=[]
for idx,w in enumerate(W):
 assert len(w['tokens'])==90
 for seed in range(3):
  table=np.zeros((5,3),dtype=float) # nchar, imvopieces, isigpieces
  for t in w['tokens']:
   b=binidx(t);table[b,0]+=sum(ch.isalpha() for ch in t)
   table[b,1]+=count(('IMVO',seed),t)
   table[b,2]+=count(('ISIG',seed),t)
  for source,col in (('IMVO',1),('ISIG',2)):
   actual=len(models[(source,seed)].encode(w['text'],out_type=int))
   if actual!=int(table[:,col].sum()):add_errors.append(dict(page=w['file'],seed=seed,model=source,actual=actual,summed=int(table[:,col].sum())))
  for b in range(5):
   row=dict(file=w['file'],issue=w['issue'],source=w['source'],seed=seed,bin=BINS[b],nchars=int(table[b,0]),IMVO_pieces=int(table[b,1]),ISIG_pieces=int(table[b,2]))
   records.append(row)

# require word count identity in all 1,242 (page, model) combinations before conclusions
assert len(add_errors)==0,('Piece additivity assumption failed',add_errors[:10])

# direct comparison to frozen Phase26 aggregate page records
old={}
with (P/'tokenizer_page_scores.csv').open(newline='',encoding='utf8') as f:
 for r in csv.DictReader(f): old[(r['file'],int(r['seed']))]=r
page=collections.defaultdict(lambda:np.zeros(3))
for r in records:
 a=page[(r['file'],r['seed'])];a[0]+=r['nchars'];a[1]+=r['IMVO_pieces'];a[2]+=r['ISIG_pieces']
max_err=0
for (file,seed),a in page.items():
 r=old[(file,seed)]
 for i,key in enumerate(('alphabetic_chars','IMVO_pieces','ISIG_pieces')):
  err=abs(a[i]-float(r[key]));max_err=max(max_err,err)
assert max_err==0,max_err

# equal-source common held-out character weights, not issue weighted
masses={s:np.zeros(5) for s in ('IMVO','ISIG')}
for r in records:
 if r['seed']==0:masses[r['source']][BINS.index(r['bin'])]+=r['nchars']
wts={s:(v/v.sum()) for s,v in masses.items()}
common=.5*(wts['IMVO']+wts['ISIG']);assert abs(sum(common)-1)<1e-8

# issue-level bin array, with all seeds; [len5, columns: chars, signed specialist-v-other piece diff]
issuecells=collections.defaultdict(lambda:np.zeros((5,2),dtype=float))
for r in records:
 s=r['source'];delta=(r['ISIG_pieces']-r['IMVO_pieces']) if s=='IMVO' else (r['IMVO_pieces']-r['ISIG_pieces'])
 a=issuecells[(s,r['seed'],r['issue'])][BINS.index(r['bin'])]
 a[0]+=r['nchars'];a[1]+=delta

def result(src,seed,issues=None):
 if issues is None: issues=sorted({iss for s,k,iss in issuecells if s==src and k==seed})
 c=np.zeros((5,2))
 for iss in issues:c+=issuecells[(src,seed,iss)]
 raw=100*c[:,1].sum()/c[:,0].sum()
 assert np.all(c[:,0]>0),(src,seed,c[:,0])
 bybin=100*c[:,1]/c[:,0]
 std=float(common@bybin)
 return float(raw),std,bybin,c

summaryrows=[]; binrows=[]
for src in ('IMVO','ISIG'):
 for seed in range(3):
  raw,std,within,c=result(src,seed)
  summaryrows.append(dict(source=src,seed=seed,raw_savings_per100=raw,length_standardized_savings_per100=std,std_minus_raw=std-raw,standardized_fraction_of_raw=std/raw if raw else None))
  for i,b in enumerate(BINS):
   binrows.append(dict(source=src,seed=seed,bin=b,source_mass_fraction=float(wts[src][i]),common_mass_fraction=float(common[i]),char_count=int(c[i,0]),per100_source_savings=float(within[i])))

rng=np.random.default_rng(29324)
cis={}
for src in ('IMVO','ISIG'):
 seeds=[]
 for _ in range(2000):
  seed=int(rng.integers(0,3))
  iss=np.array(sorted({key[2] for key in issuecells if key[0]==src and key[1]==seed}))
  draw=rng.choice(iss,len(iss),replace=True)
  raw,std,_,_=result(src,seed,draw)
  seeds.append((raw,std,std-raw))
 vals=np.array(seeds)
 cis[src]={key:[float(t) for t in np.quantile(vals[:,i],[.025,.975])] for i,key in enumerate(('raw95','standardized95','std_minus_raw95'))}

avg={s:{'raw':float(np.mean([r['raw_savings_per100'] for r in summaryrows if r['source']==s])),'standardized':float(np.mean([r['length_standardized_savings_per100'] for r in summaryrows if r['source']==s]))} for s in ('IMVO','ISIG')}
for s,d in avg.items():d['change_percent']=float(100*(d['standardized']/d['raw']-1))
res={'additivity_window_model_tests':6*len(W),'additivity_errors':len(add_errors),'max_difference_vs_phase26_page_counts':max_err,'common_character_mass_weights':dict(zip(BINS,map(float,common))),'source_character_mass_weights':{s:dict(zip(BINS,map(float,wts[s]))) for s in wts},'means':avg,'issue_seed_bootstrap_cis':cis,'gates':{
'C1_residual_positive_both':all(cis[s]['standardized95'][0]>0 for s in ('IMVO','ISIG')),
'C2_strong_mediation_both':all(avg[s]['standardized']<=.5*avg[s]['raw'] for s in ('IMVO','ISIG'))}}
(D/'LENGTH_STANDARDIZATION_RESULTS.json').write_text(json.dumps(res,indent=2))
for filename,rows in [('length_bin_savings.csv',binrows),('length_standardization_by_seed.csv',summaryrows),('length_page_bin_counts.csv',records)]:
 with (D/filename).open('w',newline='',encoding='utf8') as f:
  wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
print(json.dumps(res,indent=2))
