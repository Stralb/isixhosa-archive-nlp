#!/usr/bin/env python3
"""Phase5 secondary non-overlap diagnostic; never overrides primary F5 decision."""
import argparse, csv, json
from pathlib import Path
import numpy as np
import pandas as pd
ARMS=['PP','PW','WP','WW']; SEEDS=[441,442,443,444,445]

def contrasts(e):
    return dict(allocation=.5*((e['WP']-e['PP'])+(e['WW']-e['PW'])),
                within_view=.5*((e['PW']-e['PP'])+(e['WW']-e['WP'])),
                interaction=(e['WW']-e['WP'])-(e['PW']-e['PP']),WW_minus_PP=e['WW']-e['PP'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args()
    d=Path(a.output)
    whole=pd.read_csv(d/'heldout_lm_scores.csv')
    overlaps=pd.read_csv(d/'heldout_repetition_overlap.csv')
    assert len(whole)==99720 and len(overlaps)==2493
    allkeys=set(zip(overlaps.filename,overlaps.view))
    assert allkeys==set(zip(whole.filename,whole.view))
    df=whole.merge(overlaps[['filename','view','clean_from_train_13gram']],on=['filename','view'],validate='many_to_one')
    df=df[df.clean_from_train_13gram].copy()
    rows=[]
    for period in ('1888','1889-94'):
        # Resample same full set of issues as primary; some may have no clean pages in a particular view.
        issues=sorted(whole.loc[whole.period==period,'issue'].unique())
        n=len(issues);assert n==(52 if period=='1888' else 306)
        rng=np.random.default_rng(343401+(period!='1888'))
        wt=rng.multinomial(n,[1/n]*n,size=5000).astype(float)
        for fam in ('bpe','unigram'):
            for view in ('en','xh'):
                t=df[(df.period==period)&(df.family==fam)&(df.view==view)]
                cs=t[(t.arm=='PP')&(t.seed==441)].groupby('issue').chars.sum()
                ch=np.array([cs.get(i,0) for i in issues],dtype=float)
                nmr=t.groupby(['arm','seed','issue']).bits.sum()
                num=np.array([[[nmr.get((arm,seed,i),0.) for i in issues] for seed in SEEDS] for arm in ARMS])
                p=num.sum(axis=2)/ch.sum()
                bd=wt@ch
                assert (bd>0).all()
                bs=(wt @ num.reshape(20,n).T).reshape(5000,4,5)/bd[:,None,None]
                eff=contrasts({arm:p[k] for k,arm in enumerate(ARMS)})
                draws=contrasts({arm:bs[:,k,:] for k,arm in enumerate(ARMS)})
                for c,vals in eff.items():
                    lo,hi=np.quantile(draws[c].mean(axis=1),[.025,.975])
                    rows.append(dict(family=fam,period=period,view=view,contrast=c,
                        mean=float(np.mean(vals)),ci_low=float(lo),ci_high=float(hi),
                        seed_min=float(np.min(vals)),seed_max=float(np.max(vals)),
                        seeds_negative=int((vals<0).sum()),seeds_positive=int((vals>0).sum()),
                        clean_page_views=len(t)//20,issues_with_clean_view=t.issue.nunique(),
                        issues_in_resampling_frame=n))
    # Do not modify primary outputs; compare pre-existing separately calculated secondary estimands.
    old=pd.read_csv(d/'no_training_13gram_overlap_contrasts.csv')
    ref={(r.family,r.period,r.view,r.contrast):r.mean for r in old.itertuples()}
    assert len(ref)==32
    for r in rows:
        assert abs(r['mean']-ref[r['family'],r['period'],r['view'],r['contrast']])<1e-12
    with (d/'no_training_13gram_overlap_bootstrap.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps([r for r in rows if r['family']=='bpe' and r['view']=='en' and r['contrast']=='allocation'],indent=2))

if __name__=='__main__':main()
