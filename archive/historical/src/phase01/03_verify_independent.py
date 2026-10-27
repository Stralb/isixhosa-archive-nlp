#!/usr/bin/env python3
"""Independent, corpus-to-score audit. Does not import producer modules."""
import argparse,collections,csv,hashlib,json,re,zipfile
from pathlib import Path
import numpy as np
from scipy.stats import norm


def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()

def csvrows(path):
 with open(path,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def run(source,base,out):
 base=Path(base);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 orig=json.loads((base/'dataset_manifest.json').read_text())
 assert sha(source)==orig['source_zip_sha256'],'ZIP checksum changed'
 pages=csvrows(base/'pages.csv');features=[r['feature'] for r in csvrows(base/'features.csv')]
 npz=np.load(base/'counts.npz');Y=npz['word_counts'];N=npz['word_totals']
 assert len(pages)==292 and len(features)==200 and Y.shape==(292,200)
 assert orig['by_source']['IMVO']['issues']==49 and orig['by_source']['ISIG']['issues']==12
 with zipfile.ZipFile(source) as z:
  for i,row in enumerate(pages):
   raw=z.read(row['zip_entry'])
   assert hashlib.sha256(raw).hexdigest()==row['text_sha256'],f'raw mismatch {i}'
   words=re.findall(r'[^\W\d_]+',raw.decode('utf-8',errors='replace').casefold(),flags=re.UNICODE)
   cnt=collections.Counter(words)
   assert len(words)==int(N[i])==int(row['n_tokens']),f'token mismatch {i}'
   assert all(cnt[w]==int(Y[i,j]) for j,w in enumerate(features)),f'count mismatch {i}'
 index=collections.defaultdict(list)
 for i,r in enumerate(pages):index[(r['source'],r['issue'],r['page_half'])].append(i)
 group={}
 for k,indices in index.items():group[k]={'n':float(N[indices].sum()),'y':Y[indices].sum(axis=0).astype(float)}
 draws=[json.loads(l) for l in (base/'draws.jsonl').read_text().splitlines()]
 assert len(draws)==400
 effects=np.load(base/'all_feature_effects.npz')['effects']
 pvals=np.load(base/'all_feature_effects.npz')['pvalues']
 max_delta=0;max_pval=0
 for d in draws:
  si={'SOURCE':0,'PAGE':1}[d['scenario']];i=d['rep']
  dat={t:[(r['source'],r['issue'],r['half']) for r in d[t]] for t in ('A','B')}
  aissues={r[1] for r in dat['A']};bissues={r[1] for r in dat['B']}
  assert not aissues.intersection(bissues)
  assert (len(aissues),len(bissues))==((10,10) if si==0 else (12,12))
  for t in ('A','B'):
   ids={r[1] for r in dat[t]}
   for s in ('IMVO','ISIG'):
    count=sum(x.startswith(s+'_') for x in ids)
    if si==0:expect=(8 if t=='A' else 2) if s=='IMVO' else (2 if t=='A' else 8)
    else:expect=6
    assert count==expect,(si,t,s,count,expect)
    if si==1:
     nfront=sum(r[0]==s and r[2]=='front' for r in dat[t]);nback=sum(r[0]==s and r[2]=='back' for r in dat[t])
     assert (nfront,nback)==((4,2) if t=='A' else (2,4))
  for j,method in enumerate(('POOLED','SOURCE','SOURCE_PAGE')):
   per=[]
   for t in ('A','B'):
    keys=dat[t]
    if method=='POOLED':
     totaln=sum(group[k]['n'] for k in keys)
     totaly=sum((group[k]['y'] for k in keys),np.zeros(200))
     r=totaly/totaln;v=None
     per.append((r,v,totaln,totaly));continue
    strata=([(s,) for s in ('IMVO','ISIG')]
      if method=='SOURCE' else [(s,h) for s in ('IMVO','ISIG') for h in ('front','back')])
    rates=[];vars=[]
    for st in strata:
     groupkeys=[k for k in keys if k[0]==st[0] and (len(st)==1 or k[2]==st[1])]
     if method=='SOURCE':
      # independent construction: sum the two page halves for each issue.
      buckets=collections.defaultdict(list)
      for k in groupkeys:buckets[k[1]].append(k)
      numer=np.array([sum((group[k]['y'] for k in ks),np.zeros(200)) for ks in buckets.values()])
      denom=np.array([sum(group[k]['n'] for k in ks) for ks in buckets.values()])
     else:
      numer=np.array([group[k]['y'] for k in groupkeys])
      denom=np.array([group[k]['n'] for k in groupkeys])
     assert len(denom)>=2
     denom_total=denom.sum();mean_rate=numer.sum(axis=0)/denom_total
     cluster_resids=numer-denom[:,None]*mean_rate[None,:]
     cluster_var=(len(denom)/(len(denom)-1))*(cluster_resids**2).sum(axis=0)/denom_total**2
     rates.append(mean_rate);vars.append(cluster_var)
    per.append((sum(rates)/len(rates),sum(vars)/len(rates)**2,None,None))
   delta=per[1][0]-per[0][0]
   if method=='POOLED':
    nA,nB=per[0][2],per[1][2]
    pooled=(per[0][3]+per[1][3])/(nA+nB)
    stderr=np.sqrt(pooled*(1-pooled)*(1/nA+1/nB))
   else:stderr=np.sqrt(per[0][1]+per[1][1])
   z=np.divide(np.abs(delta),stderr,out=np.zeros_like(delta),where=stderr>0)
   p=np.clip(2*norm.sf(z),0,1)
   max_delta=max(max_delta,float(np.abs(delta-effects[si,i,j]).max()))
   max_pval=max(max_pval,float(np.abs(p-pvals[si,i,j]).max()))
 assert max_delta<1e-12,max_delta
 assert max_pval<1e-10,max_pval
 metricrows=csvrows(base/'method_comparison.csv')
 assert len(metricrows)==6
 for mr in metricrows:
  si={'SOURCE':0,'PAGE':1}[mr['scenario']];j={'POOLED':0,'SOURCE':1,'SOURCE_PAGE':2}[mr['method']]
  effects0=np.abs(effects[si,:,j,:]);v=np.median(np.median(effects0,axis=1))*10000
  assert abs(v-float(mr['median_over_reps_of_feature_median_abs_per10k']))<1e-10
 res={'status':'PASS','source_pages_verified':len(pages),'features_verified':len(features),
       'simulated_draws_verified':len(draws),'per_feature_contrasts_verified':int(effects.size),
       'pvalues_verified':int(pvals.size),'max_effect_diff':max_delta,'max_pvalue_diff':max_pval,
       'source_zip_sha256':sha(source)}
 (out/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(res,indent=2)+'\n')
 print(json.dumps(res,indent=2))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--zip',required=True);a.add_argument('--data',required=True);a.add_argument('--output',required=True)
 x=a.parse_args();run(x.zip,x.data,x.output)
