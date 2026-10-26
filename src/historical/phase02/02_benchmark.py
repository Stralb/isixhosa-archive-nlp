#!/usr/bin/env python3
"""Counterfactual historical archive missingness benchmark. NO semantic labels.
Replicate masks and all estimators on a frozen finite page-uniform target.
"""
import argparse,hashlib,json,math,warnings
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

SCENARIOS=['MAR_SOURCE_PAGE','MAR_LENGTH_DATE','MNAR_LATENT','ISSUE_DROPOUT','ZERO_CELL','REPEAT_PREF']
POINT=['POOLED','SOURCE_PAGE_STD','EXACT_CELL_MATCH','KNOWN_IPW_ORACLE','EST_IPW_FRAME']
BOUNDS=['MANSKI','GAMMA_0005','GAMMA_0020']
GAMMAS=[0.0005,0.0020]
CAP=.02
EPS=.00075
SEED=2026102411


def rng_from(*parts):
    digest=hashlib.sha256('|'.join(map(str,parts)).encode('ascii')).digest()
    return np.random.default_rng(int.from_bytes(digest[:8],'big'))


def get_data(folder,split):
    folder=Path(folder)
    frame=pd.read_csv(folder/'frame.csv')
    dat=np.load(folder/'counts.npz',allow_pickle=False)
    mask=frame.split.eq(split).to_numpy()
    fr=frame.loc[mask].reset_index(drop=True)
    counts=dat['counts'][mask].astype(np.int64)
    n=dat['total_tokens'][mask].astype(int)
    assert len(fr)==len(counts) and len(fr.issue.unique())>=10
    assert np.all(counts.sum(axis=1)<=n)
    src=(fr.source=='ISIG').astype(int).values
    half=(fr.page_half=='back').astype(int).values
    cells=2*src+half
    idx=np.array(fr.groupby('issue',sort=False).ngroup().values,dtype=int)
    return fr,counts,n,cells,idx


def injection(raw,n,split):
    """B counts modified; every page exactly retains n alphabetic positions.
    Count changes in measured forms are coherent with background substitutions.
    """
    rng=rng_from('INJECTION',SEED,split)
    b=raw.copy()
    up=np.asarray([18,24,30,36,42,48]);down=np.asarray([6,8,10,12,14,16]);
    for i in range(len(raw)):
        old=raw[i].copy()
        removed=rng.binomial(old[down],0.4)
        b[i,down]-=removed
        bg=int(n[i]-old.sum()+removed.sum())
        assert bg>=0
        k=int(rng.binomial(bg,6*0.00085))
        gain=rng.multinomial(k,np.full(6,1/6))
        b[i,up]+=gain
        assert b[i].sum()<=n[i] and np.all(b[i]>=0)
        # Exact preservation of non-target measured forms.
        unaffected=np.setdiff1d(np.arange(80),np.r_[up,down])
        assert np.array_equal(old[unaffected],b[i,unaffected])
    return b


def score_outcomes(raw,n,split):
    B=injection(raw,n,split)
    ya=np.minimum(raw/n[:,None],CAP)
    yb=np.minimum(B/n[:,None],CAP)
    truth=np.stack([np.zeros(raw.shape[1]),yb.mean(axis=0)-ya.mean(axis=0)])
    return np.stack([ya,ya]),np.stack([ya,yb]),truth,B


def sigmoid(x):return expit(x)


def propensities(scenario,frame,cells,issues):
    ns=len(frame)
    src=(frame.source=='ISIG').to_numpy().astype(float)
    back=(frame.page_half=='back').to_numpy().astype(float)
    month=frame.month.to_numpy().astype(float)
    length=frame.n_tokens.to_numpy().astype(float)
    repeated=frame.has_cross_issue_13gram.to_numpy().astype(bool)
    z=frame.hidden_text_proxy.to_numpy().astype(float)
    if scenario=='MAR_SOURCE_PAGE':
        pi=np.stack([np.array([.84,.72,.38,.34])[cells],np.array([.38,.34,.82,.72])[cells]])
    elif scenario=='MAR_LENGTH_DATE':
        length_scaled=np.clip((np.log1p(length)-np.log1p(850)), -1.5,1.5)
        base=.6*length_scaled+.55*(month-6.5)/6+.20*(src-.5)+.13*(back-.5)
        pi=np.stack([np.clip(sigmoid(.7+base),.12,.93),np.clip(sigmoid(.7-base),.12,.93)])
    elif scenario=='MNAR_LATENT':
        pi=np.stack([np.clip(sigmoid(.5+1.85*z+.2*(src-.5)),.05,.95),np.clip(sigmoid(.5-1.85*z+.2*(src-.5)),.05,.95)])
    elif scenario=='ISSUE_DROPOUT':
        piA=np.clip(.78+.035*src-.12*(month>6),.45,.9)
        piB=np.clip(.72-.14*src+.10*(month>6),.45,.9)
        pi=np.stack([piA,piB])
        # Issue-level probability should be CONSTANT across that issue.
        for g in np.unique(issues):
            assert np.allclose(pi[:,issues==g],pi[:,issues==g][:,[0]])
    elif scenario=='ZERO_CELL':
        pi=np.stack([np.array([.78,.75,.68,.68])[cells],np.array([.76,.73,.71,0.])[cells]])
    elif scenario=='REPEAT_PREF':
        pi=np.stack([np.clip(np.where(repeated,.83,.28)+.03*(src-.5),.08,.95),
                     np.clip(np.where(repeated,.28,.83)+.03*(src-.5),.08,.95)])
    else:raise ValueError(scenario)
    assert pi.shape==(2,ns) and np.all((pi>=0)&(pi<=1))
    return pi


def make_mask(scenario,frame,pi,issues,split,rep):
    rng=rng_from('SELECTION',SEED,split,scenario,rep)
    if scenario=='ISSUE_DROPOUT':
        output=np.zeros_like(pi,dtype=bool)
        for g in np.unique(issues):
            i=np.where(issues==g)[0][0]
            output[:,issues==g]=rng.random(2)[:,None]<pi[:,i,None]
        return output
    return rng.random(pi.shape)<pi


def logistic_probability(fr,selected):
    n=len(fr)
    src=(fr.source=='ISIG').to_numpy().astype(float)
    half=(fr.page_half=='back').to_numpy().astype(float)
    month=fr.month.to_numpy().astype(float)
    sz=np.log1p(fr.n_tokens.to_numpy().astype(float))
    x=np.column_stack([src,half,src*half,(month-6.5)/6, np.clip(sz-np.log1p(850),-2,2)])
    pi=np.empty_like(selected,dtype=float)
    for t in range(2):
        if selected[t].min()==selected[t].max():
            pi[t,:]=float(np.mean(selected[t]))
        else:
            model=LogisticRegression(C=1.,max_iter=150,solver='lbfgs',tol=1e-8)
            model.fit(x,selected[t].astype(int))
            pi[t,:]=model.predict_proba(x)[:,1]
    return np.clip(pi,.08,.98)


def delta_point(y,frame,cells,mask,pi,scenario,split,rep):
    """Five point methods. Return shape [M,F]."""
    n,F=y.shape[1:]
    values=np.full((len(POINT),F),np.nan)
    if min(mask.sum(axis=1))==0:return values
    obsmeans=np.array([y[t,mask[t]].mean(axis=0) for t in range(2)])
    values[0]=obsmeans[1]-obsmeans[0]
    cellsizes=np.bincount(cells,minlength=4)
    groups=np.asarray([np.where(cells==s)[0] for s in range(4)],dtype=object)
    if all(np.any(mask[t,g]) for t in range(2) for g in groups):
        means=np.stack([np.stack([y[t,g[mask[t,g]]].mean(axis=0) for g in groups]) for t in range(2)])
        values[1]=((means[1]-means[0])*cellsizes[:,None]/n).sum(axis=0)
        mrng=rng_from('MATCH',SEED,split,scenario,rep)
        matched=[]
        for t in range(2):
            parts=[]
            for s,g in enumerate(groups):
                avail=g[mask[t,g]]
                k=min(int(mask[0,g].sum()),int(mask[1,g].sum()))
                parts.append(y[t,mrng.choice(avail,size=k,replace=False)].mean(axis=0))
            matched.append((np.asarray(parts)*cellsizes[:,None]/n).sum(axis=0))
        values[2]=matched[1]-matched[0]
    if np.all(pi>0):
        ht=np.asarray([(mask[t,:,None]*y[t]/pi[t,:,None]).sum(axis=0)/n for t in range(2)])
        values[3]=ht[1]-ht[0]
    fitted=logistic_probability(frame,mask)
    weighted=[]
    for t in range(2):
        w=mask[t]/fitted[t]
        weighted.append((w[:,None]*y[t]).sum(axis=0)/w.sum())
    values[4]=weighted[1]-weighted[0]
    return values


def compute_bounds(y,cells,mask):
    """Exact Manski + assumption-conditional pattern mixture intervals.
    Returns [K,F,2], assumption premise [2,F] for gamma intervals.
    """
    n,F=y.shape[1:]
    out=np.empty((3,F,2),dtype=float)
    allidx=np.arange(n)
    lb=[];ub=[]
    for t in range(2):
        so=y[t,mask[t]].sum(axis=0)
        missing=n-int(mask[t].sum())
        lb.append(so/n);ub.append((so+missing*CAP)/n)
    out[0,:,0]=lb[1]-ub[0]
    out[0,:,1]=ub[1]-lb[0]
    holds=np.ones((2,F),dtype=bool)
    for gi,gamma in enumerate(GAMMAS):
        bounds=[]
        for t in range(2):
            a=np.zeros(F);b=np.zeros(F)
            for s in range(4):
                g=allidx[cells==s]
                included=g[mask[t,g]]
                missing=g[~mask[t,g]]
                n_obs=len(included);n_miss=len(missing)
                if n_obs:
                    mu=y[t,included].mean(axis=0)
                    misslo=np.maximum(0,mu-gamma)
                    misshi=np.minimum(CAP,mu+gamma)
                    if n_miss:
                        missing_mu=y[t,missing].mean(axis=0)
                        holds[gi] &= np.abs(missing_mu-mu)<=gamma+1e-14
                    a+=mu*n_obs+misslo*n_miss
                    b+=mu*n_obs+misshi*n_miss
                else:
                    b+=CAP*n_miss
            bounds.append((a/n,b/n))
        out[gi+1,:,0]=bounds[1][0]-bounds[0][1]
        out[gi+1,:,1]=bounds[1][1]-bounds[0][0]
    assert np.all(out[:,:,1]>=out[:,:,0]-1e-13)
    return out,holds


def main(inputs,split,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    frame,raw,n,cells,issues=get_data(inputs,split)
    nfeat=raw.shape[1];num=len(frame); reps=80 if split=='dev' else 160
    world0,world1,truth,counts_b=score_outcomes(raw,n,split)
    y=np.stack([world0,world1]) # shape world,t,page,feature
    assert np.array_equal(world0[0],world0[1]) and np.all(np.isfinite(y))
    assert np.all((y>=0)&(y<=CAP))
    pm=np.full((len(SCENARIOS),reps,2,len(POINT),nfeat),np.nan)
    bi=np.full((len(SCENARIOS),reps,2,len(BOUNDS),nfeat,2),np.nan)
    av=np.full((len(SCENARIOS),reps,2,2,nfeat),False,dtype=bool)
    masks=np.zeros((len(SCENARIOS),reps,2,num),dtype=bool)
    piarray=np.zeros((len(SCENARIOS),2,num),dtype=float)
    diag=[]
    for si,scenario in enumerate(SCENARIOS):
        pi=propensities(scenario,frame,cells,issues);piarray[si]=pi
        for r in range(reps):
            m=make_mask(scenario,frame,pi,issues,split,r)
            masks[si,r]=m
            for w in range(2):
                pm[si,r,w]=delta_point(y[w],frame,cells,m,pi,scenario,split,r)
                bi[si,r,w],av[si,r,w]=compute_bounds(y[w],cells,m)
            diag.append({'scenario':scenario,'replicate':r,
                         'observed_A_pages':int(m[0].sum()),'observed_B_pages':int(m[1].sum()),
                         'strata_A_occupied':int(len(set(cells[m[0]]))),'strata_B_occupied':int(len(set(cells[m[1]]))),
                         'zero_cell_A':int(len(set(cells[m[0]]))<4),'zero_cell_B':int(len(set(cells[m[1]]))<4),
                         'observed_A_issues':int(len(set(issues[m[0]]))),'observed_B_issues':int(len(set(issues[m[1]])))})
        print('DONE',split,scenario,'point nan fraction',round(float(np.isnan(pm[si]).mean()),3),'mean mask',round(float(masks[si].mean()),3),flush=True)
    np.savez_compressed(out/'benchmark.npz',point_effects=pm,identified_sets=bi,assumption_valid=av,
                        masks=masks,selection_probabilities=piarray,truth=truth,base_counts=raw,injected_counts=counts_b,
                        page_totals=n,cells=cells,issue_group=issues,
                        scenarios=np.array(SCENARIOS),points=np.array(POINT),bounds=np.array(BOUNDS))
    pd.DataFrame(diag).to_csv(out/'selection_diagnostics.csv',index=False)
    feature=pd.read_csv(Path(inputs)/'feature_manifest.csv',keep_default_na=False)
    assert (feature.signal=='NULL').sum()==68 and (feature.signal=='UP').sum()==6 and (feature.signal=='DOWN').sum()==6
    trs=[]
    for w,label in enumerate(['NULL','INJECTED']):
        for j,x in enumerate(truth[w]):
            trs.append({'world':label,'feature_index':j,'word_form':feature.loc[j,'word_form'],
                        'signal':feature.loc[j,'signal'],'target_change_per_10000_page_tokens':x*1e4,
                        'target_change':x})
    pd.DataFrame(trs).to_csv(out/'true_effects.csv',index=False)
    sm=[]
    sig=feature.signal.to_numpy()!='NULL'
    for si,scenario in enumerate(SCENARIOS):
        for w,label in enumerate(['NULL','INJECTED']):
            target=truth[w]
            for mi,method in enumerate(POINT):
                est=pm[si,:,w,mi,:]
                finite=np.isfinite(est)
                decision=(np.abs(est)>EPS)&finite
                corr=(np.sign(est)==np.sign(target)[None,:])&decision
                error=decision&~corr
                sm.append({'split':split,'scenario':scenario,'world':label,'method':method,'kind':'point',
                           'estimable_fraction':float(finite.mean()),
                           'mae_per_10000':float(np.nanmean(np.abs(est-target[None,:]))*1e4) if finite.any() else np.nan,
                           'decision_coverage':float(decision.mean()),'null_false_sign_fraction':float(error[:,~sig].mean()),
                           'signal_recovery':float(corr[:,sig].mean()) if w else np.nan,
                           'signal_wrong_sign_fraction':float(error[:,sig].mean()) if w else np.nan,
                           'interval_coverage':np.nan,'interval_width_per_10000':np.nan,'premise_holds':np.nan})
            for ki,method in enumerate(BOUNDS):
                lo=bi[si,:,w,ki,:,0];hi=bi[si,:,w,ki,:,1]
                inside=(lo-1e-14<=target[None,:])&(target[None,:]<=hi+1e-14)
                decision=(lo>0)|(hi<0)
                correct=decision&(np.sign(np.where(lo>0,1,-1))==np.sign(target)[None,:])
                false=decision&~correct
                sm.append({'split':split,'scenario':scenario,'world':label,'method':method,'kind':'set',
                           'estimable_fraction':1.,'mae_per_10000':np.nan,
                           'decision_coverage':float(decision.mean()),
                           'null_false_sign_fraction':float(false[:,~sig].mean()),
                           'signal_recovery':float(correct[:,sig].mean()) if w else np.nan,
                           'signal_wrong_sign_fraction':float(false[:,sig].mean()) if w else np.nan,
                           'interval_coverage':float(inside.mean()),
                           'interval_width_per_10000':float((hi-lo).mean()*1e4),
                           'premise_holds':np.nan if ki==0 else float(av[si,:,w,ki-1,:].mean())})
    pd.DataFrame(sm).to_csv(out/'summary_metrics.csv',index=False,float_format='%.13g')
    manifest={'phase':'P2','split':split,'page_count':num,'issue_count':frame.issue.nunique(),
              'feature_count':nfeat,'null_feature_count':int((~sig).sum()),'signal_feature_count':int(sig.sum()),
              'per_scenario_replicates':reps,'worlds':['NULL','INJECTED'],'scenarios':SCENARIOS,
              'point_methods':POINT,'set_methods':BOUNDS,'page_outcome_cap':CAP,'practical_sign_threshold':EPS,
              'mean_signal_abs_true_effect_per_10k':float(np.mean(np.abs(truth[1,sig]))*1e4),
              'note':'Identification-set coverage is NOT 95% sampling confidence coverage.'}
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('FINAL',split, 'truth in injected world (per 10k)',np.round(truth[1,sig]*1e4,3),'outputs',out)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',required=True);ap.add_argument('--split',choices=['dev','test'],required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();main(args.inputs,args.split,args.out)
