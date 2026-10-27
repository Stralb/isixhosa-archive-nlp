#!/usr/bin/env python3
"""Known-null archival composition stress test with strong old-style baselines.

The available 1885 issues are the finite population for this pilot; the 'time' labels
are assigned arbitrarily. Two mechanisms manipulate sampling composition only.
"""
import argparse,collections,csv,json,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import norm

METHODS=('POOLED','SOURCE','SOURCE_PAGE')
SCENARIOS=('SOURCE','PAGE')
SOURCES=('IMVO','ISIG')
HALVES=('front','back')
REPS=200
SEED=20261024


def load(data_dir):
 with open(Path(data_dir)/'pages.csv',encoding='utf-8',newline='') as f:pages=list(csv.DictReader(f))
 f=np.load(Path(data_dir)/'counts.npz');Y=f['word_counts'].astype(float);N=f['word_totals'].astype(float)
 assert Y.shape==(len(pages),200)
 keys={}
 for i,p in enumerate(pages): keys.setdefault((p['source'],p['issue'],p['page_half']),[]).append(i)
 groups=[]
 for s,issue,half in sorted(keys):
  inds=keys[(s,issue,half)]
  groups.append({'source':s,'issue':issue,'half':half,'n':N[inds].sum(),'y':Y[inds].sum(axis=0)})
 assert len(groups)==(49+12)*2
 return groups


def draw(groups,rng,scenario):
 issues={s:sorted({g['issue'] for g in groups if g['source']==s}) for s in SOURCES}
 lookup={(g['issue'],g['half']):i for i,g in enumerate(groups)}
 ans={'A':[],'B':[]}
 if scenario=='SOURCE':
  for s in SOURCES:
   v=rng.permutation(issues[s]);nA,nB=((8,2) if s=='IMVO' else (2,8))
   for label,chosen in (('A',v[:nA]),('B',v[nA:nA+nB])):
    for issue in chosen:
     for half in HALVES:ans[label].append(lookup[(issue,half)])
 elif scenario=='PAGE':
  for s in SOURCES:
   v=rng.permutation(issues[s])[:12]
   for label,chosen,nfront in [('A',v[:6],4),('B',v[6:12],2)]:
    chosen=rng.permutation(chosen)
    for i,issue in enumerate(chosen):
     ans[label].append(lookup[(issue,'front' if i<nfront else 'back')])
 else:raise ValueError(scenario)
 issueA={groups[i]['issue'] for i in ans['A']}
 issueB={groups[i]['issue'] for i in ans['B']}
 assert not issueA.intersection(issueB),(scenario,issueA.intersection(issueB))
 if scenario=='SOURCE':assert (len(issueA),len(issueB))==(10,10)
 if scenario=='PAGE':
  assert (len(issueA),len(issueB))==(12,12)
  for period in ('A','B'):
   for s in SOURCES:
    for h in HALVES:assert sum(groups[i]['source']==s and groups[i]['half']==h for i in ans[period])>=2
 return ans


def bh_count(p,alpha=0.05):
 x=np.sort(np.asarray(p,float));n=len(x)
 yes=np.flatnonzero(x<=alpha*np.arange(1,n+1)/n)
 return (int(yes[-1])+1 if yes.size else 0)


def method_eval(groups,ids,method):
 rows=[groups[i] for i in ids]
 if method=='POOLED':
  n=sum(g['n'] for g in rows);y=sum((g['y'] for g in rows),np.zeros(200,float))
  return y/n,None,n,y
 if method=='SOURCE':strata=[(s,) for s in SOURCES]
 elif method=='SOURCE_PAGE':strata=[(s,h) for s in SOURCES for h in HALVES]
 else:raise ValueError(method)
 ratios=[];variances=[]
 for st in strata:
  subset=[g for g in rows if (g['source'],) == st or (g['source'],g['half'])==st]
  if method=='SOURCE':
   grouped=collections.defaultdict(list)
   for x in subset:grouped[x['issue']].append(x)
   y=np.array([sum((x['y'] for x in ls),np.zeros(200,float)) for ls in grouped.values()])
   n=np.array([sum(x['n'] for x in ls) for ls in grouped.values()],float)
  else:
   y=np.array([x['y'] for x in subset])
   n=np.array([x['n'] for x in subset],float)
  assert len(n)>=2,(method,st,len(n))
  r=y.sum(axis=0)/n.sum()
  var=(len(n)/(len(n)-1))*np.sum((y-r[None,:]*n[:,None])**2,axis=0)/n.sum()**2
  ratios.append(r);variances.append(var)
 return np.mean(ratios,axis=0),np.sum(variances,axis=0)/len(strata)**2,None,None


def compare(groups,ans,method):
 a,va,na,ya=method_eval(groups,ans['A'],method)
 b,vb,nb,yb=method_eval(groups,ans['B'],method)
 d=b-a
 if method=='POOLED':
  p_pool=(ya+yb)/(na+nb)
  se=np.sqrt(p_pool*(1-p_pool)*(1/na+1/nb))
 else:se=np.sqrt(va+vb)
 z=np.divide(np.abs(d),se,out=np.zeros_like(d),where=se>0)
 p=np.clip(2*norm.sf(z),0,1)
 return d,p


def run(data_dir,out_dir):
 out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
 groups=load(data_dir)
 rng=np.random.default_rng(SEED)
 all_effects=np.zeros((len(SCENARIOS),REPS,len(METHODS),200),float)
 all_p=np.zeros_like(all_effects)
 draws=[];summary=[]
 for si,scenario in enumerate(SCENARIOS):
  for rep in range(REPS):
   ans=draw(groups,rng,scenario)
   draws.append({'scenario':scenario,'rep':rep,
                 'A':[{'source':groups[i]['source'],'issue':groups[i]['issue'],'half':groups[i]['half']} for i in ans['A']],
                 'B':[{'source':groups[i]['source'],'issue':groups[i]['issue'],'half':groups[i]['half']} for i in ans['B']]})
   for mi,method in enumerate(METHODS):
    d,p=compare(groups,ans,method)
    all_effects[si,rep,mi,:]=d
    all_p[si,rep,mi,:]=p
    summary.append({'scenario':scenario,'replicate':rep,'method':method,
      'median_abs_per10k':float(np.median(np.abs(d))*10000),
      'mean_abs_per10k':float(np.mean(np.abs(d))*10000),
      'BH_005_disc':bh_count(p),
      'nominal_p005_count':int(np.sum(p<0.05))})
 with open(out_dir/'replicate_metrics.csv','w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 (out_dir/'draws.jsonl').write_text(''.join(json.dumps(d,sort_keys=True)+'\n' for d in draws),encoding='utf-8')
 np.savez_compressed(out_dir/'all_feature_effects.npz',effects=all_effects,pvalues=all_p)
 report=[]
 for si,scenario in enumerate(SCENARIOS):
  base=np.median(np.abs(all_effects[si,:,0,:]),axis=1)
  for mi,method in enumerate(METHODS):
   fx=all_effects[si,:,mi,:];pv=all_p[si,:,mi,:]
   med=np.median(np.abs(fx),axis=1)
   disc=np.array([bh_count(p) for p in pv])
   report.append({'scenario':scenario,'method':method,'replicates':REPS,'features':200,
    'median_over_reps_of_feature_median_abs_per10k':float(np.median(med)*10000),
    'mean_over_reps_of_feature_median_abs_per10k':float(np.mean(med)*10000),
    'ratio_of_median_abs_to_pooled':float(np.median(med)/np.median(base)),
    'fraction_rep_median_better_than_pooled':float(np.mean(med<base)),
    'median_BH_discoveries':float(np.median(disc)),
    'mean_BH_discoveries':float(np.mean(disc)),
    'fraction_reps_any_BH_discovery':float(np.mean(disc>0)),
    'mean_nominal_p005_features':float(np.mean(np.sum(pv<.05,axis=1)))})
 with open(out_dir/'method_comparison.csv','w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(report[0]));w.writeheader();w.writerows(report)
 with open(Path(data_dir)/'features.csv',encoding='utf-8') as f:features=[r['feature'] for r in csv.DictReader(f)]
 focus=[]
 for si,scenario in enumerate(SCENARIOS):
  # Descriptive: chosen after run for largest *pooled* averaged absolute artifact.
  abs_mean=np.mean(np.abs(all_effects[si,:,0,:]),axis=0)
  for rank,i in enumerate(np.argsort(-abs_mean)[:12],start=1):
   focus.append({'scenario':scenario,'posthoc_rank':rank,'wordform':features[i],
       **{f'{METHODS[mi]}_mean_abs_per10k':float(np.mean(np.abs(all_effects[si,:,mi,i]))*10000) for mi in range(3)}})
 with open(out_dir/'descriptive_top_wordforms.csv','w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(focus[0]));w.writeheader();w.writerows(focus)
 (out_dir/'summary.json').write_text(json.dumps({'design':'1885 known-null two scenarios, 200 pseudo-archives each',
    'outcome_units':'wordform occurrence/10000 tokens, NOT word sense or semantic change',
    'methods':'POOLED token-z (invalid iid), SOURCE and SOURCE_PAGE cluster-delta z (descriptive, few issues per cell)',
    'summary':report},indent=2)+'\n',encoding='utf-8')
 print(json.dumps(report,indent=2))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--data',required=True);a.add_argument('--out',required=True)
 x=a.parse_args();run(x.data,x.out)
