#!/usr/bin/env python3
"""Independent reference-p-value, tabulation and integrity checks (not producer import)."""
import csv,json
from pathlib import Path
import numpy as np

def fdr(p):
 k=np.where(np.sort(p)<=(.05*np.arange(1,len(p)+1)/len(p)))[0]
 return 0 if not len(k) else int(k[-1])+1

def main():
 base=Path(__file__).resolve().parents[1]/'outputs'
 with (base/'oracle_comparison.csv').open() as f:rows=list(csv.DictReader(f))
 ans=[]
 for scenario in ('SOURCE','PAGE'):
  npz=np.load(base/f'oracle_{scenario.lower()}.npz')
  ref=npz['ref'];orig=npz['original_pooled'];center=npz['null_center'];p=npz['pvalues']
  assert ref.shape==(5000,200) and orig.shape==p.shape==(200,200)
  assert np.max(np.abs(ref.mean(axis=0)-center))<1e-14
  independent_p=(1+(np.abs(ref[:,None,:]-center[None,None,:])>=np.abs(orig[None,:,:]-center[None,None,:])).sum(axis=0))/5001
  assert np.array_equal(independent_p,p)
  c=np.array([fdr(row) for row in independent_p]);rr=next(x for x in rows if x['scenario']==scenario)
  for name,v in [('mean_BH_discoveries',c.mean()),('median_BH_discoveries',np.median(c)),('fraction_reps_any_BH_discovery',np.mean(c>0))]:
   assert abs(float(rr[name])-v)<1e-12,(name,rr[name],v)
  ans.append({'scenario':scenario,'reference_draws':5000,'observed_draws':200,'comparisons_recomputed':40000,
              'mean_BH_discoveries':float(c.mean()),'fraction_any':float(np.mean(c>0))})
 out=Path(__file__).resolve().parents[1]/'verification'/'ORACLE_INDEPENDENT_VERIFICATION.json'
 out.write_text(json.dumps({'status':'PASS','results':ans},indent=2)+'\n')
 print(out.read_text())

if __name__=='__main__':main()
