#!/usr/bin/env python3
"""Independent, vectorized complete Phase3 audit. No dependency on evaluator.
Recheck source text, random masks, all missing-data bounds and all non-fitted
point methods, oracle design CIs; independently refit fitted estimators in spots.
"""
import argparse,hashlib,json,re,zipfile
from collections import Counter
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression,Ridge
CAP=.02
RX=re.compile(r'[^\W\d_]+',re.UNICODE)

def shafile(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def rg(*parts):
 h=hashlib.sha256('|'.join(map(str,parts)).encode()).digest()
 return np.random.default_rng(int.from_bytes(h[:8],'big'))

def almost(a,b,label,eps=1e-10):
 x=np.asarray(a);y=np.asarray(b)
 assert x.shape==y.shape,(label,x.shape,y.shape)
 assert np.array_equal(np.isnan(x),np.isnan(y)),label+' NaN mismatch'
 good=np.isfinite(x)&np.isfinite(y)
 err=float(np.max(np.abs(x[good]-y[good]))) if good.any() else 0.
 assert err<eps,(label,err,eps)
 return err

def independent_audit(original,root,split):
 root=Path(root);source=root/'inputs'
 assert shafile(original)=='375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5'
 assert shafile(root/'PRECOMMIT.md')=='93dfb035febb58a86b6aa0b77be1bbff9eb15a6cf58699495579c52f7e966c95'
 frame=pd.read_csv(source/'frame.csv')
 ft=pd.read_csv(source/'feature_manifest.csv',keep_default_na=False)
 dat=np.load(source/'counts.npz',allow_pickle=False)
 assert len(frame)==1084 and dat['counts'].shape==(1084,80)
 with zipfile.ZipFile(original) as z:
  for i,row in enumerate(frame.itertuples()):
   raw=z.read(row.zip_entry)
   assert hashlib.sha256(raw).hexdigest()==row.text_sha256
   tokens=RX.findall(raw.decode('utf-8','replace').casefold())
   assert len(tokens)==row.n_tokens
   ct=Counter(tokens)
   assert np.array_equal(np.fromiter((ct[w] for w in ft.word_form),dtype=int),dat['counts'][i])
 print('SOURCE',split,'retokenized 1084 original pages',flush=True)
 f=frame.loc[frame.split==split].reset_index(drop=True)
 base=dat['counts'][frame.split==split];n=f.n_tokens.to_numpy()
 t=(f.period=='B').to_numpy().astype(int)
 src=(f.source=='ISIG').to_numpy().astype(int)
 half=(f.page_half=='back').to_numpy().astype(int)
 cell=2*src+half
 issue,unique=pd.factorize(f.issue,sort=False)
 d=np.load(root/'outputs'/split/'benchmark.npz',allow_pickle=False)
 for k,v in [('base_counts',base),('page_totals',n),('period',t),('cells',cell),('issue_group',issue)]:
  assert np.array_equal(d[k],v),k
 mod=base.copy();gen=rg('P3-INJECTED',20261024,split)
 for i in np.flatnonzero(t==1):
  down=np.array([6,8,10,12,14,16]);up=np.array([18,24,30,36,42,48])
  taken=gen.binomial(mod[i,down],.4)
  mod[i,down]-=taken
  bg=int(n[i]-base[i].sum()+taken.sum());val=int(gen.binomial(bg,.0051))
  mod[i,up]+=gen.multinomial(val,np.full(6,1/6))
 assert np.array_equal(d['injected_counts'],mod)
 Y=np.stack([np.minimum(base/n[:,None],CAP),np.minimum(mod/n[:,None],CAP)])
 truth=np.asarray([y[t==1].mean(axis=0)-y[t==0].mean(axis=0) for y in Y])
 almost(truth,d['truth'],'truth',1e-13)
 met=json.loads((source/'frame_manifest.json').read_text())
 x=np.column_stack([src,half,src*half,(f.month.to_numpy()-6.5)/6,
                    np.clip(np.log1p(n)-np.log1p(1000),-2.5,2.5)])
 R=d['masks'].shape[1]
 totals={'source_pages':1084,'replicated_random_flags':0,'all_manski_bounds':0,'all_gamma_bounds':0,
         'all_tipping_grids':0,'all_pooled_std_known_ht_point_values':0,'all_oracle_design_ci_values':0,
         'all_gamma_truth_premises':0,'fitted_model_spot_estimates':0}
 largest=0.
 scenarios=[str(s) for s in d['scenarios']]
 for si,scenario in enumerate(scenarios):
  for w in range(2):
   y=Y[w]
   month=f.month.to_numpy().astype(float);size=n.astype(float)
   repeated=f.has_cross_issue_13gram.to_numpy().astype(bool)
   if scenario=='MAR_SOURCE_PAGE':p=np.where(t==0,np.array([.84,.70,.38,.34])[cell],np.array([.38,.34,.82,.72])[cell])
   elif scenario=='MAR_LENGTH_DATE':
    z=np.clip(np.log1p(size)-np.log1p(1000),-2.5,2.5)
    a=.6*z+.55*(month-6.5)/6+.20*(src-.5)+.13*(half-.5)
    p=np.clip(expit(.7+np.where(t==0,a,-a)),.12,.93)
   elif scenario=='MNAR_LATENT':
    z=np.clip((y[:,:4].sum(axis=1)-met['dev_hidden_proxy_mean'])/met['dev_hidden_proxy_sd'],-2.5,2.5)
    p=np.clip(expit(.5+np.where(t==0,1.85*z,-1.85*z)+.2*(src-.5)),.05,.95)
   elif scenario=='ISSUE_DROPOUT':p=np.where(t==0,np.clip(.78+.035*src-.12*(month>6),.45,.9),np.clip(.72-.14*src+.1*(month>6),.45,.9))
   elif scenario=='ZERO_CELL':p=np.where(t==0,np.array([.78,.75,.68,.68])[cell],np.array([.76,.73,.71,0.])[cell])
   elif scenario=='REPEAT_PREF':p=np.clip(np.where(t==0,np.where(repeated,.83,.28),np.where(repeated,.28,.83))+.03*(src-.5),.08,.95)
   largest=max(largest,almost(p,d['selection_probabilities'][si,w],scenario+' probabilities',1e-12))
   observed=np.empty((R,len(n)),bool)
   for r in range(R):
    random=rg('P3-MASK',20261024,split,scenario,r,w)
    if scenario=='ISSUE_DROPOUT':
     draw=np.zeros(len(n),bool)
     for g in np.unique(issue):
      ix=issue==g;draw[ix]=random.random()<p[ix][0]
    else:draw=random.random(len(n))<p
    observed[r]=draw
   assert np.array_equal(observed,d['masks'][si,:,w])
   totals['replicated_random_flags']+=observed.size
   period_sums=[];period_upper=[];period_mean=[];period_std=[];period_ht=[];period_htvariance=[]
   # Precalculate per-cell observed/missing means for exact gamma bounds at all masks.
   cs={}
   for per in (0,1):
    ix=t==per;Np=int(ix.sum());m=observed[:,ix].astype(float); yy=y[ix];pi=p[ix]
    obsum=m@yy;counts=m.sum(axis=1)
    period_sums.append(obsum/Np)
    period_upper.append((obsum+(Np-counts)[:,None]*CAP)/Np)
    period_mean.append(np.divide(obsum,counts[:,None],out=np.full((R,80),np.nan),where=counts[:,None]>0))
    # Standardize to actual period-specific population cell mix, not 50/50 sources.
    val=np.zeros((R,80));valid=np.ones(R,bool)
    for c in range(4):
     row=ix&(cell==c);nn=int(row.sum());ii=observed[:,row].astype(float);cc=ii.sum(axis=1)
     so=ii@y[row]
     mm=np.divide(so,cc[:,None],out=np.zeros((R,80)),where=cc[:,None]>0)
     val+=(nn/Np)*mm
     valid &= cc>0
     cs[per,c]=(nn,cc,so,y[row].sum(axis=0))
    val[~valid]=np.nan
    period_std.append(val)
    if (pi>0).all():
     ht=m@(yy/pi[:,None])/Np
     if scenario!='ISSUE_DROPOUT':
      var=m@((1-pi)[:,None]*yy**2/pi[:,None]**2)/(Np**2)
     else:
      group=issue[ix];uni=np.unique(group);irows=np.array([np.flatnonzero(group==g)[0] for g in uni],dtype=object)
      # Each issue is one Bernoulli observation: outcome totals carry covariance.
      vari=np.zeros((R,80))
      for g in uni:
       members=group==g;v=yy[members].sum(axis=0)
       pg=pi[members][0];sampled=m[:,members][:,0]
       vari+=sampled[:,None]*(1-pg)/pg**2*v[None,:]**2/(Np**2)
      var=vari
    else:ht=np.full((R,80),np.nan);var=np.full((R,80),np.nan)
    period_ht.append(ht);period_htvariance.append(var)
   pt=d['point_effects'][si,:,w]
   for j,e in ((0,period_mean[1]-period_mean[0]),(1,period_std[1]-period_std[0]),(5,period_ht[1]-period_ht[0])):
    largest=max(largest,almost(e,pt[:,j],f'{split} {scenario} point {j}'))
    totals['all_pooled_std_known_ht_point_values']+=e.size
   val=period_ht[1]-period_ht[0]
   se=np.sqrt(period_htvariance[0]+period_htvariance[1])
   oracle=np.stack([val-1.96*se,val+1.96*se],axis=2)
   largest=max(largest,almost(oracle,d['point_ci'][si,:,w,3],scenario+'oracle_ci'))
   totals['all_oracle_design_ci_values']+=oracle.size
   lower=period_sums[1]-period_upper[0];upper=period_upper[1]-period_sums[0]
   mb=np.stack([lower,upper],axis=2)
   largest=max(largest,almost(mb,d['identified_sets'][si,:,w,0],scenario+' Manski'))
   assert np.all(lower<truth[w][None,:]+1e-12) and np.all(upper>truth[w][None,:]-1e-12)
   totals['all_manski_bounds']+=lower.size
   def gamma_bounds(gamma):
    lows=[];ups=[];premise=np.ones((R,80),bool)
    for per in (0,1):
     Np=int((t==per).sum());lo=np.zeros((R,80));hi=np.zeros((R,80))
     for c in range(4):
      nn,cn,so,popsum=cs[per,c]
      missing=nn-cn
      mu=np.divide(so,cn[:,None],out=np.zeros((R,80)),where=cn[:,None]>0)
      lowmissing=np.maximum(0,mu-gamma)
      highmissing=np.where(cn[:,None]>0,np.minimum(CAP,mu+gamma),0.)
      lo+=so+missing[:,None]*lowmissing
      hi+=so+missing[:,None]*highmissing
      hi+=(cn==0)[:,None]*missing[:,None]*CAP
      missingmu=np.divide(popsum[None,:]-so,missing[:,None],out=np.zeros((R,80)),where=missing[:,None]>0)
      premise &= ((cn==0)|(missing==0))[:,None] | (np.abs(missingmu-mu)<=gamma+1e-14)
     lows.append(lo/Np);ups.append(hi/Np)
    return np.stack([lows[1]-ups[0],ups[1]-lows[0]],axis=2),premise
   for k,gamma in enumerate((.0005,.0020)):
    val,premise=gamma_bounds(gamma)
    largest=max(largest,almost(val,d['identified_sets'][si,:,w,k+1],scenario+' gamma'))
    assert np.array_equal(premise,d['gamma_premise_holds'][si,:,w,k])
    totals['all_gamma_bounds']+=R*80;totals['all_gamma_truth_premises']+=premise.size
   tipping=np.full((R,80),np.nan)
   for gamma in (0.,.00025,.0005,.001,.002,.004,.01):
    val,_=gamma_bounds(gamma)
    hits=np.isnan(tipping)&(val[:,:,0]<=0)&(val[:,:,1]>=0)
    tipping[hits]=gamma
   largest=max(largest,almost(tipping,d['tipping_gamma'][si,:,w],scenario+' tipping'))
   totals['all_tipping_grids']+=R*80
   # Independently refit propensity and ridge outcome models for 3 deterministic masks.
   for r in (0,R//2,R-1):
    ms=observed[r];fittedp=np.zeros(len(n));pred=np.zeros_like(y)
    for per in (0,1):
     ix=t==per;obs=ix&ms
     if np.unique(ms[ix]).size<2:fittedp[ix]=np.clip(ms[ix].mean(),.08,.98)
     else:
      clf=LogisticRegression(C=1,max_iter=100,solver='lbfgs',tol=1e-7).fit(x[ix],ms[ix].astype(int))
      fittedp[ix]=clf.predict_proba(x[ix])[:,1]
     fittedp[ix]=np.clip(fittedp[ix],.08,.98)
     if obs.sum()<8:pred[ix]=y[obs].mean(axis=0) if obs.any() else 0
     else:
      mdl=Ridge(alpha=6.,fit_intercept=True).fit(x[obs],y[obs]);pred[ix]=np.clip(mdl.predict(x[ix]),0,CAP)
    parts=[[],[],[]]
    for per in (0,1):
     ix=t==per;obs=ix&ms;weights=1/fittedp[obs]
     parts[0].append((weights[:,None]*y[obs]).sum(axis=0)/weights.sum())
     parts[1].append(pred[ix].mean(axis=0))
     adj=pred[ix].copy();adj[ms[ix]]+=(y[obs]-pred[obs])/fittedp[obs,None]
     parts[2].append(adj.mean(axis=0))
    for j,idx in enumerate((2,3,4)):
     largest=max(largest,almost(parts[j][1]-parts[j][0],pt[r,idx],scenario+' fitspot'))
     totals['fitted_model_spot_estimates']+=80
   print('CHECKED',split,scenario,'world',w,flush=True)
 result={'status':'PASS','split':split,'original_source_sha256':shafile(original),
         'phase3_protocol_sha256':shafile(root/'PRECOMMIT.md'),
         'max_numerical_abs_difference':largest,'metrics':totals,
         'interpretation':'Complete-frame finite-population Manski bounds exact, only known-HT CI has design-based basis when true inclusion probabilities known; fitted model CIs are heuristic.'}
 (root/'outputs'/split/'INDEPENDENT_VECTOR_AUDIT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print('VERIFIED',split,json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--root',required=True);p.add_argument('--split',choices=['dev','test'],required=True);a=p.parse_args()
 independent_audit(a.source,a.root,a.split)
