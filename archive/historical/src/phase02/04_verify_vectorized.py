#!/usr/bin/env python3
"""Fast independent full-arithmetic verification of Phase2 outcomes.
Uses independent vectorized formulas, original raw ZIP, and spot checks for nonclosed-form methods.
"""
import argparse, collections, hashlib, json,re,zipfile
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
RXP=re.compile(r'[^\W\d_]+')
CAP=.02

def srng(*parts):
 m='|'.join(map(str,parts)).encode('ascii');return np.random.default_rng(int.from_bytes(hashlib.sha256(m).digest()[:8],'big'))

def check(in_dir,out_dir,rawzip,split):
 frame=pd.read_csv(Path(in_dir)/'frame.csv'); features=pd.read_csv(Path(in_dir)/'feature_manifest.csv',keep_default_na=False)
 assert features.signal.value_counts().to_dict()=={'NULL':68,'UP':6,'DOWN':6}
 orig=np.load(Path(in_dir)/'counts.npz',allow_pickle=False)
 z=np.load(Path(out_dir)/'benchmark.npz',allow_pickle=False)
 rows=frame.split==split;fr=frame.loc[rows].reset_index(drop=True)
 C=orig['counts'][rows.to_numpy()];nt=orig['total_tokens'][rows.to_numpy()];B=z['injected_counts']
 assert np.array_equal(C,z['base_counts']) and np.array_equal(nt,z['page_totals'])
 assert B.shape==C.shape and (B>=0).all() and (B.sum(axis=1)<=nt).all()
 assert np.array_equal(B[:,features.signal=='NULL'],C[:,features.signal=='NULL'])
 assert len(set(fr.issue)&set(frame.loc[frame.split=='train','issue']))==0
 assert len(set(frame.loc[frame.split=='test','issue'])&set(frame.loc[frame.split=='dev','issue']))==0
 # Check entire original source text again, including every simulated hidden page.
 raw_sha=hashlib.sha256(Path(rawzip).read_bytes()).hexdigest()
 assert raw_sha=='375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5'
 with zipfile.ZipFile(rawzip) as source:
  for i,row in frame.iterrows():
   txt=source.read(row.zip_entry)
   assert hashlib.sha256(txt).hexdigest()==row.text_sha256
   words=RXP.findall(txt.decode('utf8','replace').casefold());ct=collections.Counter(words)
   assert len(words)==orig['total_tokens'][i]
   assert np.array_equal(np.array([ct[x] for x in features.word_form]),orig['counts'][i])
 print('RAW_ARCHIVE',len(frame),'pages and 80 forms verified',flush=True)
 A=np.minimum(C/nt[:,None],CAP);BB=np.minimum(B/nt[:,None],CAP)
 Y=np.stack([np.stack([A,A]),np.stack([A,BB])]) # (W,T,P,F)
 truth=np.stack([np.zeros(80),BB.mean(axis=0)-A.mean(axis=0)])
 np.testing.assert_allclose(z['truth'],truth,atol=1e-14,rtol=0)
 assert np.all(truth[1,features.signal=='UP']>0) and np.all(truth[1,features.signal=='DOWN']<0)
 assert np.all(truth[1,features.signal=='NULL']==0)
 M=z['masks'].astype(float);pi=z['selection_probabilities'];S,R,T,P=M.shape
 assert (S,T,P)==(6,2,len(fr))
 groups=(fr.source.eq('ISIG').astype(int)*2+fr.page_half.eq('back').astype(int)).to_numpy()
 assert np.array_equal(groups,z['cells'])
 maskcounts=M.sum(axis=-1) # (S,R,T)
 # (S,R,W,T,F) sum of observed outcomes
 sums=np.einsum('srtp,wtpf->srwtf',M,Y,optimize=True)
 pooled=np.divide(sums,maskcounts[:,:,None,:,None],out=np.full_like(sums,np.nan),where=maskcounts[:,:,None,:,None]>0)
 est_pooled=pooled[:,:,:,1,:]-pooled[:,:,:,0,:]
 np.testing.assert_allclose(est_pooled,z['point_effects'][:,:,:,0,:],rtol=0,atol=1e-12,equal_nan=True)
 if np.all(pi[0]>0):pass
 # Known-propensity HT; off-support scenario must be unestimable.
 for si in range(S):
  if (pi[si]==0).any():
   assert np.isnan(z['point_effects'][si,:,:,3,:]).all()
  else:
   w=M[si]/pi[si][None,:,:]
   aggregate=np.einsum('rtp,wtpf->rwtf',w,Y,optimize=True)/P
   np.testing.assert_allclose(aggregate[:,:,1,:]-aggregate[:,:,0,:],z['point_effects'][si,:,:,3,:],atol=1e-12,rtol=0,equal_nan=True)
 # Full poststratification and both pattern-mixture sensitivity intervals.
 std=np.zeros_like(est_pooled)
 g_l=np.zeros((2,S,R,2,2,80));g_u=np.zeros_like(g_l)
 holds=np.ones((2,S,R,2,2,80),dtype=bool)
 occupancy=np.ones((S,R,2),dtype=bool)
 for c in range(4):
  flag=(groups==c).astype(float);nc=int(flag.sum())
  mc=M*flag[None,None,None,:]
  count=mc.sum(axis=-1)
  cell_sum=np.einsum('srtp,wtpf->srwtf',mc,Y,optimize=True)
  avg=np.divide(cell_sum,count[:,:,None,:,None],out=np.zeros_like(cell_sum),where=count[:,:,None,:,None]>0)
  occupancy &=count>0
  # Average over target 4-cell proportions, but NaN on insufficient support.
  std+=(avg[:,:,:,1,:]-avg[:,:,:,0,:])*nc/P
  missing=nc-count
  missmc=(1-M)*flag[None,None,None,:]
  missing_sum=np.einsum('srtp,wtpf->srwtf',missmc,Y,optimize=True)
  missmean=np.divide(missing_sum,missing[:,:,None,:,None],out=np.zeros_like(missing_sum),where=missing[:,:,None,:,None]>0)
  for k,gamma in enumerate([.0005,.0020]):
   lo=np.where(count[:,:,None,:,None]>0,np.maximum(0,avg-gamma),0.)
   hi=np.where(count[:,:,None,:,None]>0,np.minimum(CAP,avg+gamma),CAP)
   g_l[k]+=(avg*count[:,:,None,:,None]+lo*missing[:,:,None,:,None])/P
   g_u[k]+=(avg*count[:,:,None,:,None]+hi*missing[:,:,None,:,None])/P
   fail=(np.abs(missmean-avg)>gamma+1e-14)&(missing[:,:,None,:,None]>0)&(count[:,:,None,:,None]>0)
   holds[k]&=~fail
 std[~np.all(occupancy,axis=-1)]=np.nan
 np.testing.assert_allclose(std,z['point_effects'][:,:,:,1,:],rtol=0,atol=1e-12,equal_nan=True)
 # Manski: full known finite frame, no sampling uncertainty.
 lb=sums/P;ub=lb+(P-maskcounts[:,:,None,:,None])*CAP/P
 manski=np.stack([lb[:,:,:,1,:]-ub[:,:,:,0,:],ub[:,:,:,1,:]-lb[:,:,:,0,:]],axis=-1)
 np.testing.assert_allclose(manski,z['identified_sets'][:,:,:,0,:,:],atol=1e-12,rtol=0)
 assert np.all(manski[...,0]<=truth[None,None,:,:]+1e-12)
 assert np.all(manski[...,1]>=truth[None,None,:,:]-1e-12)
 for k in range(2):
  calc=np.stack([g_l[k,:,:,:,1,:]-g_u[k,:,:,:,0,:],g_u[k,:,:,:,1,:]-g_l[k,:,:,:,0,:]],axis=-1)
  np.testing.assert_allclose(calc,z['identified_sets'][:,:,:,k+1,:,:],atol=1e-12,rtol=0)
  # holds shape (S,R,W,T,F) -> both periods and across all nonempty cells.
  keep=np.all(holds[k],axis=3)
  assert np.array_equal(keep,z['assumption_valid'][:,:,:,k,:])
  target=truth[None,None,:,:]
  inside=(calc[...,0]<=target+1e-12)&(calc[...,1]>=target-1e-12)
  assert np.all(inside[keep]),'If sensitivity assumption holds, true effect must be included.'
 print('VECTORIZED',split,'all scenarios, masks, 80 word forms, both worlds, pooled, standardization, oracle HT and 3 bound types',flush=True)
 # Independent mask replay ALL trials; metadata/condition sanity.
 for si,scenario in enumerate(z['scenarios'].tolist()):
  for r in range(R):
   rng=srng('SELECTION',2026102411,split,scenario,r)
   if scenario=='ISSUE_DROPOUT':
    expected=np.zeros((2,P),dtype=bool)
    for group in np.unique(z['issue_group']):
     ids=np.flatnonzero(z['issue_group']==group)
     expected[:,ids]=(rng.random(2)[:,None]<pi[si,:,ids[0],None])
   else: expected=rng.random((2,P))<pi[si]
   assert np.array_equal(expected,z['masks'][si,r])
  if scenario=='ZERO_CELL':assert not z['masks'][si,:,1,groups==3].any()
 print('MASKS',S*R*2*P,'all deterministic observations validated',flush=True)
 # Spot-check matching + estimated propensity against independently derived data for r=0,1,2 only.
 spotchecked=0
 X=np.column_stack([fr.source.eq('ISIG').to_numpy().astype(float),fr.page_half.eq('back').to_numpy().astype(float),
                    ((fr.source=='ISIG')&(fr.page_half=='back')).to_numpy().astype(float),
                    (fr.month.to_numpy()-6.5)/6,np.clip(np.log1p(nt)-np.log1p(850),-2,2)])
 for si,scenario in enumerate(z['scenarios'].tolist()):
  for r in range(min(3,R)):
   m=z['masks'][si,r]
   pfit=[]
   for t in range(2):
    if len(np.unique(m[t]))<2: q=np.full(P,m[t].mean())
    else:
     clf=LogisticRegression(C=1.,max_iter=150,solver='lbfgs',tol=1e-8)
     clf.fit(X,m[t].astype(int));q=clf.predict_proba(X)[:,1]
    pfit.append(np.clip(q,.08,.98))
   for world in range(2):
    mean=[]
    for t in range(2):
     weights=m[t]/pfit[t]
     mean.append(np.sum(weights[:,None]*Y[world,t],axis=0)/sum(weights))
    np.testing.assert_allclose(z['point_effects'][si,r,world,4],mean[1]-mean[0],atol=1e-11,rtol=0)
    if all(np.any(m[t,groups==g]) for t in range(2) for g in range(4)):
     rng=srng('MATCH',2026102411,split,scenario,r)
     mm=[]
     for t in range(2):
      sub=[]
      for g in range(4):
       idc=np.flatnonzero(groups==g)
       selected=idc[m[t,idc]]
       nmin=min(m[0,idc].sum(),m[1,idc].sum())
       draw=rng.choice(selected,size=int(nmin),replace=False)
       sub.append(Y[world,t,draw].mean(axis=0))
      mm.append(sum(sub[g]*(groups==g).sum()/P for g in range(4)))
     np.testing.assert_allclose(z['point_effects'][si,r,world,2],mm[1]-mm[0],atol=1e-12,rtol=0)
    spotchecked+=1
 print('SPOTCHECKED',spotchecked,'fitted-propensity and matched feature contrast vectors',flush=True)
 # Verify summary records, coverage and decision fractions for all outcomes.
 d=pd.read_csv(Path(out_dir)/'summary_metrics.csv',keep_default_na=False)
 null=(features.signal=='NULL').to_numpy();signals=~null
 for row in d.itertuples():
  i=z['scenarios'].tolist().index(row.scenario);w=['NULL','INJECTED'].index(row.world)
  if row.kind=='point':
   j=z['points'].tolist().index(row.method)
   v=z['point_effects'][i,:,w,j];finite=np.isfinite(v);decision=(np.abs(v)>.00075)&finite
   assert abs(finite.mean()-row.estimable_fraction)<1e-12
   assert abs(decision.mean()-row.decision_coverage)<1e-12
   err=decision&(np.sign(v)!=np.sign(truth[w])[None,:])
   assert abs(err[:,null].mean()-row.null_false_sign_fraction)<1e-12
  else:
   j=z['bounds'].tolist().index(row.method)
   bounds=z['identified_sets'][i,:,w,j]
   inside=(bounds[...,0]<=truth[w][None,:]+1e-14)&(truth[w][None,:]<=bounds[...,1]+1e-14)
   assert abs(inside.mean()-float(row.interval_coverage))<1e-12
   assert abs((bounds[...,1]-bounds[...,0]).mean()*1e4-float(row.interval_width_per_10000))<1e-9
 print('SUMMARY',len(d),'rows independently verified',flush=True)
 report={'status':'PASS','split':split,'raw_pages_independently_tokenized':len(frame),'heldout_pages':P,
         'masks_replayed':int(S*R*2*P),'full_closed_form_methods': ['POOLED','SOURCE_PAGE_STD','KNOWN_IPW_ORACLE','MANSKI','GAMMA_0005','GAMMA_0020'],
         'point_vectors_and_set_endpoints_checked':int(np.prod(z['identified_sets'].shape)+np.prod(z['point_effects'][:,:,:,:2,:].shape)+np.prod(z['point_effects'][:,:,:,3:4,:].shape)),
         'fitted_and_match_spotchecks':spotchecked,'all_summary_metrics_checked':len(d),
         'unrestricted_bounds_cover_truth':True,'sensitivity_bounds_cover_truth_when_assumptions_hold':True}
 (Path(out_dir)/'INDEPENDENT_VECTOR_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--inputs',required=True);ap.add_argument('--outputs',required=True);ap.add_argument('--zip',required=True);ap.add_argument('--split',required=True,choices=['dev','test']);a=ap.parse_args()
 check(a.inputs,a.outputs,a.zip,a.split)
