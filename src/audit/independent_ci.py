#!/usr/bin/env python3
"""Data-free independent validation of all 84 reported means/bootstrap intervals.

Does NOT reuse any original Phase 1-6 analysis functions, nor claim to retrain models.
Run: python src/audit/independent_ci.py --root .
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

SEEDS=(441,442,443,444,445)
ARMS=('PP','PW','WP','WW')
PERIODS=('1888','1889-94')

def contrasts(v):
    return {'allocation':((v['WP']-v['PP'])+(v['WW']-v['PW']))/2,
            'within_view':((v['PW']-v['PP'])+(v['WW']-v['WP']))/2,
            'interaction':(v['WW']-v['WP'])-(v['PW']-v['PP']),
            'WW_minus_PP':v['WW']-v['PP']}

def verify(root,raw,summary,metric,repetition=False):
    scores=pd.read_csv(root/raw)
    published=pd.read_csv(root/summary).fillna(0)
    if repetition:
        assert len(scores)==49860
        scores['family']='bpe'
        scores['arm']=scores.condition+'_'+scores.arm
        policies=('filtered_PP','filtered_WW','support_control_PP','support_control_WW')
    else:
        assert len(scores)==99720
        policies=ARMS
    assert scores.unknown.eq(0).all()
    errors=[];counter=0
    for period in PERIODS:
        sub=scores[scores.period==period]
        issues=sorted(sub.issue.unique());n=len(issues)
        assert n==(52 if period=='1888' else 306)
        samples=np.random.default_rng(343401+(period!='1888')).multinomial(n,[1/n]*n,size=5000)
        yrs=np.array([int(x[5:9]) for x in issues])
        if period!='1888':
            years=range(1889,1895)
            year_sample=np.random.default_rng(343501).multinomial(6,[1/6]*6,size=5000)
        for family in ('bpe',) if repetition else ('bpe','unigram'):
          for view in ('en','xh'):
            group=sub[(sub.family==family)&(sub.view==view)]
            reference=group[(group.arm==policies[0])&(group.seed==SEEDS[0])]
            char=np.array([reference.loc[reference.issue==issue,'chars'].sum() for issue in issues],dtype=float)
            assert char.sum()>0
            numer=np.zeros((len(policies),5,n))
            for i,arm in enumerate(policies):
              for j,seed in enumerate(SEEDS):
                x=group[(group.arm==arm)&(group.seed==seed)]
                assert len(x)==len(reference)
                amounts=x.groupby('issue')[metric].sum()
                numer[i,j,:]=[amounts.get(issue,0) for issue in issues]
            observed=numer.sum(axis=2)/char.sum()
            draws=(samples@numer.reshape(-1,n).T).reshape(5000,len(policies),5)/(samples@char)[:,None,None]
            if metric=='pieces':observed*=100;draws*=100
            if repetition:
                fpp,fww,cpp,cww=observed
                fb,fw,cb,cw=[draws[:,i,:] for i in range(4)]
                obs={'filtered_PP_minus_WW':fpp-fww,'support_control_PP_minus_WW':cpp-cww,
                     'filtered_minus_support_control':fpp-fww-cpp+cww,
                     'filtered_PP_minus_control_PP':fpp-cpp,'filtered_WW_minus_control_WW':fww-cww}
                boots={'filtered_PP_minus_WW':fb-fw,'support_control_PP_minus_WW':cb-cw,
                       'filtered_minus_support_control':fb-fw-cb+cw,
                       'filtered_PP_minus_control_PP':fb-cb,'filtered_WW_minus_control_WW':fw-cw}
            else:
                obs=contrasts(dict(zip(policies,observed)))
                boots=contrasts({arm:draws[:,i,:] for i,arm in enumerate(policies)})
            yearboots={}
            if period!='1888':
                cy=np.array([char[yrs==y].sum() for y in years])
                ny=np.stack([numer[:,:,yrs==y].sum(-1) for y in years],axis=-1)
                ydraws=(year_sample@ny.reshape(-1,6).T).reshape(5000,len(policies),5)/(year_sample@cy)[:,None,None]
                if metric=='pieces':ydraws*=100
                if repetition:
                    fb,fw,cb,cw=[ydraws[:,i,:] for i in range(4)]
                    yearboots={'filtered_PP_minus_WW':fb-fw,'support_control_PP_minus_WW':cb-cw,
                               'filtered_minus_support_control':fb-fw-cb+cw,
                               'filtered_PP_minus_control_PP':fb-cb,'filtered_WW_minus_control_WW':fw-cw}
                else:yearboots=contrasts({arm:ydraws[:,i,:] for i,arm in enumerate(policies)})
            for name,values in obs.items():
                q=published[(published.period==period)&(published.view==view)&(published.contrast==name)]
                if not repetition:q=q[q.family==family]
                assert len(q)==1,(family,period,view,name)
                row=q.iloc[0]
                computed=[float(np.mean(values)),*map(float,np.quantile(np.mean(boots[name],axis=1),[.025,.975]))]
                reported=[row['mean'],row['ci_low'],row['ci_high']]
                errors.extend(abs(a-b) for a,b in zip(computed,reported))
                assert (values<0).sum()==row['seed_negative' if repetition else 'seeds_negative']
                assert (values>0).sum()==row['seed_positive' if repetition else 'seeds_positive']
                if period!='1888':
                    year_ci=np.quantile(np.mean(yearboots[name],axis=1),[.025,.975])
                    fields=('year_ci_low','year_ci_high') if repetition else ('year_cluster_ci_low','year_cluster_ci_high')
                    errors.extend(abs(float(z)-float(row[f])) for z,f in zip(year_ci,fields))
                counter+=1
    assert counter==(20 if repetition else 32),counter
    assert max(errors)<1e-8,max(errors)
    print('PASS',summary,counter,'contrasts, max absolute error',max(errors))
    return counter

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'))
    r=p.parse_args().root
    assert (r/'results/models').is_dir()
    pairs=0
    for mf in ('model_manifest.csv','repetition_models.csv'):
        rows=pd.read_csv(r/'results'/mf)
        for x in rows.itertuples():
            for suffix in ('.model','.vocab'):
                assert (r/'results/models'/(x.model_name+suffix)).is_file()
            pairs+=1
    assert pairs==60
    total=0
    total+=verify(r,'results/raw/segmentation.csv','results/segmentation.csv','pieces')
    total+=verify(r,'results/raw/bigram.csv','results/bigram.csv','bits')
    total+=verify(r,'results/raw/repetition.csv','results/repetition.csv','pieces',True)
    assert total==84
    print('PASS: 84 contrasts and paired issue/year bootstrap intervals, 60 saved model pairs present')
if __name__=='__main__':main()
