#!/usr/bin/env python3
"""Phase4 precommitted outcome-blind simulated page-audit value of information.

No new algorithm and NO new historical pages recovered; the private oracle's masked
word counts are revealed solely as explicitly labelled counterfactual audit outcomes.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

Q=np.array([0.,.05,.10,.25,.50,.75,.90,1.])
POLICIES=['RANDOM','WIDTH_GREEDY','CELL_REPAIR']
EXPECTED={'dev':'47551f5db490029f563b77833f4d0eaabd0ea4ce5dbdd43771b821348c95a809',
          'test':'67b2dec56e8ad7dfc87896aa0742604afac25a60b83df02c9a84de92cacc68c9'}
FRAME_SHA='f3ec7f8c954b0152258020a05ee400aa6db1d17f6d865da8d89bbf5f48c6a19a'
CAP=.02
MIN_TRUTH=.0002

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def seed(*args):
 b=hashlib.sha256('|'.join(map(str,args)).encode('utf-8')).digest()
 return int.from_bytes(b[:8],'big')

def order_blind(m,period,cells,policy,seed_id):
 rg=np.random.default_rng(seed_id)
 missing=np.flatnonzero(~m)
 perm=rg.permutation(missing)
 if policy=='RANDOM':return perm
 n0=(period==0).sum();n1=(period==1).sum()
 # Sorted by mathematical per-page contribution to interval width: CAP/N_t.
 priorit=np.where(period[perm]==0,CAP/n0,CAP/n1)
 greedy=perm[np.argsort(-priorit,kind='stable')]
 if policy=='WIDTH_GREEDY':return greedy
 if policy!='CELL_REPAIR':raise ValueError(policy)
 # Empty-source/page-half cells must be repaired using only observable page IDs.
 cover=[];left=set(map(int,perm))
 for t in range(2):
  for cell in range(4):
   ix=(period==t)&(cells==cell)
   if ix.any() and not np.any(m[ix]):
    candidates=[int(i) for i in perm if ix[i] and int(i) in left]
    assert candidates
    chosen=candidates[0]
    cover.append(chosen);left.remove(chosen)
 return np.array(cover+[int(i) for i in greedy if int(i) in left],dtype=int)

def data_load(phase3,split):
 fname=Path(phase3)/'outputs'/split/'benchmark.npz';assert sha(fname)==EXPECTED[split]
 frame_path=Path(phase3)/'inputs'/'frame.csv';assert sha(frame_path)==FRAME_SHA
 with np.load(fname,allow_pickle=False) as z: data={k:z[k] for k in z.files}
 fr=pd.read_csv(frame_path);fr=fr[fr['split']==split].reset_index(drop=True)
 assert len(fr)==len(data['period'])
 assert np.array_equal(fr['n_tokens'],data['page_totals'])
 assert np.array_equal(fr['period'].map({'A':0,'B':1}),data['period'])
 assert np.array_equal(fr['page_half'].map({'front':0,'back':1})+2*fr['source'].map({'IMVO':0,'ISIG':1}),data['cells'])
 assert data['masks'].shape[:3]==(6,80 if split=='dev' else 120,2)
 return fname,data,fr

def eval_split(phase3,split,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 inputfile,data,fr=data_load(phase3,split)
 scen=[str(x) for x in data['scenarios']]
 masks=data['masks'];num_s,num_rep,num_w,num_pages=masks.shape;nf=80
 nt=data['page_totals'].astype(np.float64);period=data['period'];cells=data['cells']
 N=np.array([(period==t).sum() for t in (0,1)],dtype=np.float64)
 counts=np.stack([data['base_counts'],data['injected_counts']])
 YY=np.minimum(counts/nt[None,:,None],CAP)
 t_actual=np.asarray([y[period==1].mean(axis=0)-y[period==0].mean(axis=0) for y in YY])
 assert np.allclose(t_actual,data['truth'],rtol=0,atol=1e-14)
 bounds_store=np.full((num_s,num_rep,num_w,len(POLICIES),len(Q),nf,2),np.nan,dtype=np.float64)
 recovered=np.zeros((num_s,num_rep,num_w,len(POLICIES),len(Q)),dtype=np.int16)
 gap_count=np.zeros((num_s,num_rep,num_w,len(POLICIES),len(Q)),dtype=np.int8)
 records=[];order_records=[]
 for si,s in enumerate(scen):
  for rep in range(num_rep):
   for w in range(num_w):
    m=masks[si,rep,w].astype(bool);Y=YY[w];missing=np.flatnonzero(~m);nmiss=len(missing)
    for pi,policy in enumerate(POLICIES):
     oo=order_blind(m,period,cells,policy,seed('P4-RECOVERY',20261024,split,s,rep,w,policy))
     assert len(oo)==nmiss and len(set(map(int,oo)))==nmiss
     assert set(oo)==set(missing)
     oh=hashlib.sha256(oo.astype('<i4').tobytes()).hexdigest()
     order_records.append({'scenario':s,'replicate':rep,'world':w,'policy':policy,'order_sha256':oh,'original_missing':nmiss})
     # Audit observations accumulate as nested prefixes for each policy.
     xx=np.where(period[oo]==0,0,1)
     cs=[];cc=[]
     for t in (0,1):
      zz=Y[oo]*((xx==t)[:,None])
      cs.append(np.vstack([np.zeros((1,nf)),np.cumsum(zz,axis=0)]))
      cc.append(np.concatenate([[0],np.cumsum(xx==t)]))
     start=[Y[m & (period==t)].sum(axis=0) for t in (0,1)]
     initially_uncovered=sum(bool(np.any(period==t)&np.any(cells==c)) and not np.any(m & (period==t) & (cells==c)) for t in (0,1) for c in range(4))
     for qi,q in enumerate(Q):
      k=nmiss if q==1 else int(np.floor(q*nmiss))
      known_a=start[0]+cs[0][k];known_b=start[1]+cs[1][k]
      left_a=N[0]-((m&(period==0)).sum()+cc[0][k]);left_b=N[1]-((m&(period==1)).sum()+cc[1][k])
      lo=known_b/N[1]-(known_a+CAP*left_a)/N[0]
      hi=(known_b+CAP*left_b)/N[1]-known_a/N[0]
      assert (lo<=hi+1e-11).all()
      true=t_actual[w]
      assert np.all((lo<=true+1e-12)&(hi>=true-1e-12))
      if k==0:
       phase3b=data['identified_sets'][si,rep,w,0]
       assert np.allclose(np.column_stack((lo,hi)),phase3b,atol=2e-14,rtol=0)
      if k==nmiss:
       assert max(np.max(np.abs(lo-true)),np.max(np.abs(hi-true)))<3e-13
      decided=(lo>MIN_TRUTH)|(hi<-MIN_TRUTH)
      assert not np.any(decided & (np.sign(lo+hi)!=np.sign(true)))
      masknow=m.copy();masknow[oo[:k]]=True
      uncovered=sum(bool(np.any((period==t)&(cells==c))) and not np.any(masknow & (period==t) & (cells==c)) for t in (0,1) for c in range(4))
      gap_count[si,rep,w,pi,qi]=uncovered
      bounds_store[si,rep,w,pi,qi,:,0]=lo
      bounds_store[si,rep,w,pi,qi,:,1]=hi
      recovered[si,rep,w,pi,qi]=k
      records.append({'scenario':s,'replicate':rep,'world':['NATURAL','INJECTED'][w],
        'policy':policy,'budget':float(q),'missing_initial':nmiss,'audited_pages':k,
        'bound_width_per_10k':float(np.mean(hi-lo)*1e4),
        'practical_sign_coverage':float(np.mean(decided)),
        'practical_sign_count':int(decided.sum()),
        'practical_truth_count':int((np.abs(true)>MIN_TRUTH).sum()),
        'truth_containment':float(np.mean((lo<=true+1e-12)&(hi>=true-1e-12))),
        'wrong_calls':int(np.count_nonzero(decided & (np.sign(lo+hi)!=np.sign(true)))),
        'empty_cells_before':initially_uncovered,'empty_cells_after':uncovered})
   # print periodic checkpoints for monitoring
  print(split,s,'done',flush=True)
 result=pd.DataFrame.from_records(records)
 assert len(result)==num_s*num_rep*num_w*len(POLICIES)*len(Q)
 result.to_csv(out/'audit_replicate_results.csv',index=False,float_format='%.14g')
 order_df=pd.DataFrame(order_records);order_df.to_csv(out/'audit_orders_hashes.csv',index=False)
 np.savez_compressed(out/'bounds_and_audits.npz',lower_upper=bounds_store,audited_counts=recovered,
                     empty_cells_remaining=gap_count,budgets=Q,policies=np.asarray(POLICIES),scenarios=np.asarray(scen),truth=t_actual)
 summary=result.groupby(['scenario','world','policy','budget'],sort=True).agg(
   repetitions=('replicate','count'),mean_missing=('missing_initial','mean'),mean_audited=('audited_pages','mean'),
   mean_width_10k=('bound_width_per_10k','mean'),mean_sign_coverage=('practical_sign_coverage','mean'),
   positive_call_replicate_pct=('practical_sign_count',lambda x:100*np.mean(x>0)),
   mean_empty_cells=('empty_cells_after','mean'),
   bound_truth_containment=('truth_containment','mean'),total_wrong_calls=('wrong_calls','sum')).reset_index()
 summary.to_csv(out/'scenario_summary.csv',index=False,float_format='%.14g')
 pooled=result.groupby(['world','policy','budget'],sort=True).agg(
   n_trials=('replicate','count'),mean_width_10k=('bound_width_per_10k','mean'),
   mean_sign_coverage=('practical_sign_coverage','mean'),mean_missing=('missing_initial','mean'),
   total_wrong_calls=('wrong_calls','sum')).reset_index()
 pooled.to_csv(out/'pooled_summary.csv',index=False,float_format='%.14g')
 meta={'split':split,'phase3_benchmark_sha256':sha(inputfile),'phase3_frame_sha256':sha(Path(phase3)/'inputs'/'frame.csv'),
       'analysis':'existing Manski bounds + simulated re-discovery of masked pages; NOT actual recovery',
       'period_pages':{'A':int(N[0]),'B':int(N[1])},
       'pages':num_pages,'scenarios':scen,'replicates':num_rep,'worlds':2,'wordforms':nf,
       'audits':POLICIES,'budgets':Q.tolist(),'mask_evaluations':len(result),
       'zero_wrong_calls':bool((result.wrong_calls==0).all()),
       'full_recovery_exact':True,'original_masks_match_phase3_manski':True}
 (out/'RUN_MANIFEST.json').write_text(json.dumps(meta,sort_keys=True,indent=2)+'\n')
 print('OUTPUT',split,meta['mask_evaluations'],'rows',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase3',default='/mnt/data/Historical_Inference_Programme/Phase3');p.add_argument('--split',choices=('dev','test'),required=True);p.add_argument('--out',required=True);a=p.parse_args();eval_split(a.phase3,a.split,a.out)
