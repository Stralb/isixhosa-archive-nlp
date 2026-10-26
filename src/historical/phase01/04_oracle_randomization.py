#!/usr/bin/env python3
"""Post-hoc exact-selection-model oracle stress test (known-null only)."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from importlib.machinery import SourceFileLoader

B=5000
SEED=20261025
SCENARIOS=('SOURCE','PAGE')

def bh(p):
 q=np.sort(p);sel=np.flatnonzero(q<=.05*(1+np.arange(len(q)))/len(q))
 return int(sel[-1]+1) if len(sel) else 0

def run(dataset,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 # Reuse the EXACT frozen selection mechanism; independent verification re-evaluates the oracle statistic.
 s=SourceFileLoader('frozen_stress_test',str(Path(__file__).parent/'02_stress_test.py')).load_module()
 groups=s.load(dataset)
 empirical=np.load(Path(dataset)/'all_feature_effects.npz')['effects']
 rng=np.random.default_rng(SEED)
 rows=[];nulls=[]
 for idx,scenario in enumerate(SCENARIOS):
  ref=np.zeros((B,200),dtype=float)
  for b in range(B):
   dr=s.draw(groups,rng,scenario)
   def prevalence(ids):
    n=sum(groups[k]['n'] for k in ids)
    y=sum((groups[k]['y'] for k in ids),np.zeros(200))
    return y/n
   ref[b,:]=prevalence(dr['B'])-prevalence(dr['A'])
  nulls.append(ref)
  mean=ref.mean(axis=0)
  ref_dev=np.abs(ref-mean)
  obs_dev=np.abs(empirical[idx,:,0,:]-mean[None,:])
  p=(1+np.sum(ref_dev[:,None,:]>=obs_dev[None,:,:],axis=0))/(B+1)
  disc=np.array([bh(a) for a in p])
  rows.append({'scenario':scenario,'oracle_null_draws':B,'original_replicates':len(disc),
               'median_BH_discoveries':float(np.median(disc)),
               'mean_BH_discoveries':float(np.mean(disc)),
               'fraction_reps_any_BH_discovery':float(np.mean(disc>0)),
               'pvalue_granularity':1/(B+1),'median_feature_pvalue':float(np.median(p))})
  np.savez_compressed(out/('oracle_'+scenario.lower()+'.npz'),ref=ref,pvalues=p,original_pooled=empirical[idx,:,0,:],null_center=mean)
 with open(out/'oracle_comparison.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (out/'oracle_summary.json').write_text(json.dumps({'exploratory_after_first_pilot':True,'complete_archive_and_true_selection_rule_assumed':True,'results':rows},indent=2)+'\n')
 print(json.dumps(rows,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True)
 a=p.parse_args();run(a.data,a.out)
