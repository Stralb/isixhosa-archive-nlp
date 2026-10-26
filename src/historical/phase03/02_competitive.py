#!/usr/bin/env python3
"""Phase3 real-year archive-loss benchmark. Existing statistical methods only."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression,Ridge

SCENARIOS=['MAR_SOURCE_PAGE','MAR_LENGTH_DATE','MNAR_LATENT','ISSUE_DROPOUT','ZERO_CELL','REPEAT_PREF']
METHODS=['POOLED','SOURCE_HALF_STD','EST_IPW','RIDGE_IMPUTE','AIPW','KNOWN_HT_ORACLE']
CI_METHODS=['POOLED','SOURCE_HALF_STD','EST_IPW','KNOWN_HT_ORACLE']
BOUND_NAMES=['MANSKI','GAMMA_0005','GAMMA_0020']
GAMMAS=[.0005,.0020]
TIPPING_GAMMAS=[0.,.00025,.0005,.001,.002,.004,.01]
CAP=.02
CALL_THRESHOLD=.00075
MIN_TRUTH=.00020

def rng(*things):
 h=hashlib.sha256('|'.join(map(str,things)).encode('utf-8')).digest()
 return np.random.default_rng(int.from_bytes(h[:8],'big'))

def load(folder,split):
 folder=Path(folder);fr=pd.read_csv(folder/'frame.csv')
 a=np.load(folder/'counts.npz',allow_pickle=False)
 idx=(fr.split==split).to_numpy();fr=fr.loc[idx].reset_index(drop=True)
 counts=a['counts'][idx].astype(int);nt=a['n_tokens'][idx].astype(int)
 assert np.array_equal(nt,fr.n_tokens.to_numpy())
 periods=np.where(fr.period=='B',1,0)
 src=np.where(fr.source=='ISIG',1,0)
 back=(fr.page_half=='back').to_numpy().astype(int)
 cells=2*src+back
 issues,_=pd.factorize(fr.issue,sort=False)
 X=np.column_stack([src,back,src*back,(fr.month.to_numpy()-6.5)/6,
                     np.clip(np.log1p(nt)-np.log1p(1000),-2.5,2.5)])
 with open(folder/'frame_manifest.json') as f: manifest=json.load(f)
 return fr,counts,nt,periods,cells,issues,X,manifest

def worlds(counts,nt,period,split):
 modified=counts.copy();B=np.where(period==1)[0]
 rg=rng('P3-INJECTED',20261024,split)
 downs=np.array([6,8,10,12,14,16]);ups=np.array([18,24,30,36,42,48])
 for i in B:
  removed=rg.binomial(modified[i,downs],.4)
  modified[i,downs]-=removed
  bg=int(nt[i]-counts[i].sum()+removed.sum())
  k=int(rg.binomial(bg,.0051))
  modified[i,ups]+=rg.multinomial(k,np.full(6,1/6))
  assert modified[i].sum()<=nt[i] and (modified[i]>=0).all()
 Y=np.stack([np.minimum(counts/nt[:,None],CAP),np.minimum(modified/nt[:,None],CAP)])
 truth=np.asarray([y[period==1].mean(axis=0)-y[period==0].mean(axis=0) for y in Y])
 return Y,truth,modified

def probabilities(s,fr,period,cells,issues,Y,meta):
 N=len(fr);p=np.zeros(N)
 src=(fr.source=='ISIG').to_numpy().astype(float);back=(fr.page_half=='back').to_numpy().astype(float)
 month=fr.month.to_numpy().astype(float);length=fr.n_tokens.to_numpy().astype(float)
 rep=fr.has_cross_issue_13gram.to_numpy().astype(bool)
 if s=='MAR_SOURCE_PAGE':
  p=np.where(period==0,np.array([.84,.70,.38,.34])[cells],np.array([.38,.34,.82,.72])[cells])
 elif s=='MAR_LENGTH_DATE':
  z=np.clip(np.log1p(length)-np.log1p(1000),-2.5,2.5)
  base=.6*z+.55*(month-6.5)/6+.20*(src-.5)+.13*(back-.5)
  p=np.clip(expit(.7+np.where(period==0,base,-base)),.12,.93)
 elif s=='MNAR_LATENT':
  z=np.clip((Y[:,:4].sum(axis=1)-meta['dev_hidden_proxy_mean'])/meta['dev_hidden_proxy_sd'],-2.5,2.5)
  p=np.clip(expit(.5+np.where(period==0,1.85*z,-1.85*z)+.2*(src-.5)),.05,.95)
 elif s=='ISSUE_DROPOUT':
  p=np.where(period==0,np.clip(.78+.035*src-.12*(month>6),.45,.9),
                        np.clip(.72-.14*src+.10*(month>6),.45,.9))
  for g in np.unique(issues):assert np.allclose(p[issues==g],p[issues==g][0])
 elif s=='ZERO_CELL':
  p=np.where(period==0,np.array([.78,.75,.68,.68])[cells],np.array([.76,.73,.71,0.])[cells])
 elif s=='REPEAT_PREF':
  p=np.clip(np.where(period==0,np.where(rep,.83,.28),np.where(rep,.28,.83))+.03*(src-.5),.08,.95)
 else:raise ValueError(s)
 assert (p>=0).all() and (p<=1).all()
 return p

def mask_draw(p,s,split,rep,w,issues):
 g=rng('P3-MASK',20261024,split,s,rep,w)
 if s=='ISSUE_DROPOUT':
  m=np.zeros(len(p),bool)
  for q in np.unique(issues):
   idx=np.where(issues==q)[0]
   m[idx]=g.random()<p[idx[0]]
  return m
 return g.random(len(p))<p

def fit_prop(X,period,m):
 out=np.zeros(len(m),float)
 for t in (0,1):
  ix=period==t
  if len(np.unique(m[ix]))<2:out[ix]=max(.08,min(.98,float(m[ix].mean())))
  else:
   lm=LogisticRegression(C=1,max_iter=100,solver='lbfgs',tol=1e-7)
   lm.fit(X[ix],m[ix].astype(int))
   out[ix]=lm.predict_proba(X[ix])[:,1]
 return np.clip(out,.08,.98)

def ridge_predictions(X,period,m,Y):
 out=np.zeros_like(Y)
 for t in (0,1):
  obs=(period==t)&m
  if obs.sum()<8:out[period==t]=Y[obs].mean(axis=0) if obs.any() else 0.
  else:
   reg=Ridge(alpha=6.,fit_intercept=True)
   reg.fit(X[obs],Y[obs])
   out[period==t]=np.clip(reg.predict(X[period==t]),0,CAP)
 return out

def group_var(contrib,groups):
 # Cluster linearized variance of a ratio/stratified estimate; heuristic finite-frame CI.
 ug=np.unique(groups)
 if len(ug)<=1:return np.full(contrib.shape[1],np.nan)
 # Fast indexed issue aggregation; repeated masks are not independent issues.
 inv=np.searchsorted(ug,groups)
 z=np.zeros((len(ug),contrib.shape[1]),float)
 np.add.at(z,inv,contrib)
 return len(ug)/(len(ug)-1)*np.sum(z*z,axis=0)

def point_and_intervals(Y,fr,period,cells,groups,X,m,p,s):
 N,F=Y.shape
 val=np.full((len(METHODS),F),np.nan)
 interval=np.full((len(CI_METHODS),F,2),np.nan)
 pi_hat=fit_prop(X,period,m)
 yhat=ridge_predictions(X,period,m,Y)
 means=np.full((2,len(METHODS),F),np.nan)
 variances=np.full((2,len(CI_METHODS),F),np.nan)
 for t in (0,1):
  ix=(period==t);obs=ix&m;nt=int(ix.sum());nobs=int(obs.sum())
  if nobs<1:continue
  yy=Y[obs];gg=groups[ix]
  mu=yy.mean(axis=0);means[t,0]=mu
  c=np.zeros((nt,F));c[m[ix]]=(yy-mu)/nobs
  variances[t,0]=group_var(c,gg)
  # Standardized cell means with period-specific full-frame weights.
  if all(np.any(obs & (cells==cell)) for cell in range(4)):
   c=np.zeros((nt,F));total=np.zeros(F)
   for cell in range(4):
    row=(ix)&(cells==cell);incl=obs&(cells==cell);nm=int(incl.sum());popn=int(row.sum())
    mu_cell=Y[incl].mean(axis=0);cw=popn/nt
    total+=cw*mu_cell
    c[incl[ix]]=cw*(Y[incl]-mu_cell)/nm
   means[t,1]=total
   variances[t,1]=group_var(c,gg)
  # estimated inverse propensity Hajek
  w=1/pi_hat[obs];wm=w.sum();mu=(w[:,None]*yy).sum(axis=0)/wm
  means[t,2]=mu
  c=np.zeros((nt,F));c[m[ix]]=w[:,None]*(yy-mu)/wm
  variances[t,2]=group_var(c,gg)
  means[t,3]=yhat[ix].mean(axis=0)
  contrib=yhat[ix].copy();contrib[m[ix]]+=(Y[obs]-yhat[obs])/pi_hat[obs,None]
  means[t,4]=contrib.mean(axis=0)
  # oracle HT, design variance for finite population independent Bernoulli or issue selection
  if np.all(p[ix]>0):
   means[t,5]=(Y[obs]/p[obs,None]).sum(axis=0)/nt
   if s!='ISSUE_DROPOUT':
    var=(Y[obs]**2*((1-p[obs])/p[obs]**2)[:,None]).sum(axis=0)/(nt*nt)
   else:
    var=np.zeros(F)
    for g in np.unique(gg):
     r=(groups==g)&ix
     if m[r][0]:
      pu=p[r][0]
      totals=Y[r].sum(axis=0)
      var+=(1-pu)/pu**2*(totals/nt)**2
   variances[t,3]=var
 delta=means[1]-means[0]
 val[:]=delta.T if False else delta.T.T
 for j,kind in enumerate(CI_METHODS):
  method=METHODS.index(kind);est=val[method]
  v=variances[0,j]+variances[1,j]
  se=np.sqrt(np.maximum(v,0))
  interval[j,:,0]=est-1.96*se
  interval[j,:,1]=est+1.96*se
 return val,interval

def bounds(Y,period,cells,m):
 N,F=Y.shape
 sets=np.zeros((len(BOUND_NAMES),F,2));valid=np.ones((len(GAMMAS),F),dtype=bool)
 lo=[];hi=[]
 for t in (0,1):
  ix=period==t;obs=ix&m;npop=ix.sum();nmiss=(ix&~m).sum()
  lo.append(Y[obs].sum(axis=0)/npop)
  hi.append((Y[obs].sum(axis=0)+CAP*nmiss)/npop)
 sets[0,:,0]=lo[1]-hi[0];sets[0,:,1]=hi[1]-lo[0]
 for j,gamma in enumerate(GAMMAS):
  lx=[];ux=[]
  for t in (0,1):
   Np=(period==t).sum();lower=np.zeros(F);upper=np.zeros(F)
   for cell in range(4):
    ix=(period==t)&(cells==cell);obs=ix&m;missing=ix&~m
    no=obs.sum();nm=missing.sum()
    if no:
     mu=Y[obs].mean(axis=0)
     if nm:
      valid[j] &= np.abs(Y[missing].mean(axis=0)-mu)<=gamma+1e-14
     lower+=no*mu+nm*np.maximum(mu-gamma,0)
     upper+=no*mu+nm*np.minimum(mu+gamma,CAP)
    else:upper+=nm*CAP
   lx.append(lower/Np);ux.append(upper/Np)
  sets[j+1,:,0]=lx[1]-ux[0];sets[j+1,:,1]=ux[1]-lx[0]
 assert np.all(sets[:,:,0]<=sets[:,:,1]+1e-12)
 return sets,valid

def tipping(Y,period,cells,m):
 # minimum gamma at which delta pattern-mixture bound includes zero (grid, not exact point).
 out=np.full(Y.shape[1],np.nan)
 for gamma in TIPPING_GAMMAS:
  lx=[];ux=[]
  for t in (0,1):
   Np=(period==t).sum();lower=np.zeros(Y.shape[1]);upper=lower.copy()
   for cell in range(4):
    ix=(period==t)&(cells==cell);obs=ix&m;missing=ix&~m
    no=obs.sum();nm=missing.sum()
    if no:
     mu=Y[obs].mean(axis=0)
     lower+=no*mu+nm*np.maximum(0,mu-gamma)
     upper+=no*mu+nm*np.minimum(CAP,mu+gamma)
    else:upper+=nm*CAP
   lx.append(lower/Np);ux.append(upper/Np)
  dl=lx[1]-ux[0];du=ux[1]-lx[0]
  eligible=np.isnan(out)&(dl<=0)&(du>=0)
  out[eligible]=gamma
 return out

def eval_summary(pred,ints,sets,val,truth):
 F=len(truth);true_direct=np.abs(truth)>MIN_TRUTH
 finite=np.isfinite(pred)
 dec=finite&(np.abs(pred)>CALL_THRESHOLD)
 correct=dec&true_direct[None,None,None,:]&(np.sign(pred)==np.sign(truth)[None,None,None,:])
 wrong=dec&~correct
 return correct,wrong,dec

def summaries(split,data,out):
 pred=data['point_effects'];ints=data['point_ci'];sets=data['identified_sets'];truth=data['truth'];valid=data['gamma_premise_holds']
 S,R,W,M,F=pred.shape
 names=[str(x) for x in data['scenarios']]
 rows=[];curve=[]
 for si,s in enumerate(names):
  for w,label in enumerate(('NATURAL','INJECTED')):
   yy=truth[w]
   real=np.abs(yy)>MIN_TRUTH
   for mi,method in enumerate(METHODS):
    est=pred[si,:,w,mi,:];finite=np.isfinite(est)
    dec=(np.abs(est)>CALL_THRESHOLD)&finite
    correct=dec&real[None,:]&(np.sign(est)==np.sign(yy)[None,:])
    wrong=dec&~correct
    r={'split':split,'scenario':s,'world':label,'method':method,'class':'point',
       'mean_absolute_error_per_10000':float(np.nanmean(np.abs(est-yy[None,:]))*1e4) if finite.any() else np.nan,
       'estimable_fraction':float(finite.mean()),'decision_coverage':float(dec.mean()),
       'wrong_call_rate_among_decisions':float(wrong.sum()/max(1,dec.sum())),
       'correct_action_recall':float(correct.sum()/max(1,real.sum()*R)),
       'effect_in_interval':np.nan,'interval_width_per_10000':np.nan,'premise_holds':np.nan}
    if method in CI_METHODS:
     k=CI_METHODS.index(method);ci=ints[si,:,w,k,:,:]
     within=(ci[:,: ,0]<=yy[None,:])&(yy[None,:]<=ci[:,:,1])
     fin_ci=np.isfinite(ci).all(axis=2)
     r['effect_in_interval']=float(within[fin_ci].mean()) if fin_ci.any() else np.nan
     r['interval_width_per_10000']=float(np.nanmean(ci[:,:,1]-ci[:,:,0])*1e4) if fin_ci.any() else np.nan
     r['ci_sign_call_fraction']=float(((ci[:,:,0]>0)|(ci[:,:,1]<0)).mean())
    rows.append(r)
    for fraction in (.10,.20,.30):
     k=int(round(F*fraction));n_wrong=0;n_dec=0;n_corr=0;n_reps=0
     for rep in range(R):
      ix=np.where(finite[rep])[0]
      if len(ix)<k:continue
      order=ix[np.argsort(-np.abs(est[rep,ix]),kind='stable')[:k]]
      verdict=np.sign(est[rep,order])
      is_ok=real[order]&(verdict==np.sign(yy[order]))
      n_dec+=len(order);n_wrong+=int((~is_ok).sum());n_corr+=int(is_ok.sum());n_reps+=1
     curve.append({'split':split,'scenario':s,'world':label,'method':method,'requested_coverage':fraction,
                   'achieved_coverage':n_dec/(R*F),'wrong_call_rate':n_wrong/n_dec if n_dec else np.nan,
                   'correct_count':n_corr,'eligible_reps':n_reps})
   for ki,method in enumerate(BOUND_NAMES):
    bb=sets[si,:,w,ki,:,:];lo=bb[:,:,0];hi=bb[:,:,1]
    dec=(lo>0)|(hi<0)
    corr=dec&real[None,:]&(np.where(lo>0,1,-1)==np.sign(yy)[None,:])
    wrong=dec&~corr
    inside=(lo<=yy[None,:])&(yy[None,:]<=hi)
    rows.append({'split':split,'scenario':s,'world':label,'method':method,'class':'identification_set',
      'mean_absolute_error_per_10000':np.nan,'estimable_fraction':1.,'decision_coverage':float(dec.mean()),
      'wrong_call_rate_among_decisions':float(wrong.sum()/max(1,dec.sum())),
      'correct_action_recall':float(corr.sum()/max(1,real.sum()*R)),
      'effect_in_interval':float(inside.mean()),'interval_width_per_10000':float((hi-lo).mean()*1e4),
      'premise_holds':np.nan if ki==0 else float(valid[si,:,w,ki-1,:].mean())})
 return pd.DataFrame(rows),pd.DataFrame(curve)

def main(inp,split,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 fr,cnt,nt,period,cells,groups,X,meta=load(inp,split)
 Y,truth,modified=worlds(cnt,nt,period,split)
 N=len(fr);F=cnt.shape[1];R=80 if split=='dev' else 120;S=len(SCENARIOS);W=2;M=len(METHODS)
 point=np.full((S,R,W,M,F),np.nan);pci=np.full((S,R,W,len(CI_METHODS),F,2),np.nan)
 bs=np.full((S,R,W,len(BOUND_NAMES),F,2),np.nan);prem=np.zeros((S,R,W,2,F),bool)
 tip=np.full((S,R,W,F),np.nan);mask=np.zeros((S,R,W,N),bool);pstore=np.zeros((S,W,N),float)
 for si,s in enumerate(SCENARIOS):
  for w in range(W):pstore[si,w]=probabilities(s,fr,period,cells,groups,Y[w],meta)
  for r in range(R):
   for w in range(W):
    p=pstore[si,w];mm=mask_draw(p,s,split,r,w,groups);mask[si,r,w]=mm
    point[si,r,w],pci[si,r,w]=point_and_intervals(Y[w],fr,period,cells,groups,X,mm,p,s)
    bs[si,r,w],prem[si,r,w]=bounds(Y[w],period,cells,mm)
    tip[si,r,w]=tipping(Y[w],period,cells,mm)
  print('FINISHED',split,s,'mean observed',round(mask[si].mean(),4),flush=True)
 np.savez_compressed(out/'benchmark.npz',point_effects=point,point_ci=pci,identified_sets=bs,
   gamma_premise_holds=prem,tipping_gamma=tip,masks=mask,selection_probabilities=pstore,
   truth=truth,base_counts=cnt,injected_counts=modified,page_totals=nt,period=period,cells=cells,issue_group=groups,
   scenarios=np.array(SCENARIOS),methods=np.array(METHODS),ci_methods=np.array(CI_METHODS),bound_methods=np.array(BOUND_NAMES))
 sm,curves=summaries(split,{'point_effects':point,'point_ci':pci,'identified_sets':bs,
                   'gamma_premise_holds':prem,'truth':truth,'scenarios':np.array(SCENARIOS)},out)
 sm.to_csv(out/'summary_metrics.csv',index=False,float_format='%.12g');curves.to_csv(out/'risk_coverage.csv',index=False,float_format='%.12g')
 feat=pd.read_csv(Path(inp)/'feature_manifest.csv',keep_default_na=False)
 pd.DataFrame([{'world':label,'index':j,'word_form':feat.iloc[j].word_form,
                'injected_status':feat.iloc[j].engineered_status,'full_archive_delta':float(truth[w,j]),
                'full_archive_delta_per_10000':float(truth[w,j]*1e4)} for w,label in enumerate(['NATURAL','INJECTED']) for j in range(F)]).to_csv(out/'true_effects.csv',index=False)
 (out/'run_manifest.json').write_text(json.dumps({'schema':'PRHI_PHASE3_ADJACENT_YEARS_V1',
  'split':split,'years':sorted(fr.year.unique().tolist()),'pages':N,'issues':int(fr.issue.nunique()),
  'repetitions_per_scenario':R,'scenario_names':SCENARIOS,'features':F,'methods':METHODS,
  'bound_methods':BOUND_NAMES,'ci_methods':CI_METHODS,'alpha_ci':.05,'practical_call_threshold':CALL_THRESHOLD,
  'minimal_true_effect_for_correct_call':MIN_TRUTH,'cap':CAP,
  'important':'Natural-world lexical effects are not zero; synthetic injection is not validated semantic change.'},indent=2)+'\n')
 print('TARGET',split,'abs true median rate per 10k',np.median(np.abs(truth[0])*1e4),'total',N,'features',F)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--inputs',required=True);ap.add_argument('--split',choices=['dev','test'],required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
 main(a.inputs,a.split,a.out)
