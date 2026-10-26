#!/usr/bin/env python3
"""Paper-readable plots from frozen Phase5 output tables; never changes estimates."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args()
    out=Path(a.output)
    d=pd.read_csv(out/'bpc_factorial_contrasts.csv')
    m=d[d.contrast=='allocation'].copy()
    rows=[]
    for period in ('1888','1889-94'):
        for view in ('en','xh'):
            for family in ('bpe','unigram'):
                rows.append(m[(m.period==period)&(m.view==view)&(m.family==family)].iloc[0])
    fig,ax=plt.subplots(figsize=(10,5.6))
    for i,r in enumerate(rows):
        x=float(r['mean']);lo=float(r['ci_low']);hi=float(r['ci_high']);y=7-i
        ax.errorbar(x,y,xerr=[[x-lo],[hi-x]],fmt=('o' if r['family']=='bpe' else 's'),capsize=3,
                     markersize=6,alpha=.85)
    ax.axvline(0,linestyle='--',linewidth=1,color='.45')
    ax.axvline(-.01,linestyle=':',linewidth=1,color='.55')
    ax.set_yticks(list(range(7,-1,-1)))
    ax.set_yticklabels([f"{r['period']}  |  {r['view'].upper()}  |  {r['family'].upper()}" for r in rows])
    ax.set_xlabel('LM loss effect: word-like minus page-like language allocation (bits/character)')
    ax.set_title('Phase 5: downstream language-allocation contrasts\n95% paired publication-issue bootstrap intervals')
    ax.grid(axis='x',alpha=.2)
    fig.text(.02,.015,'Negative = better prediction under word-like allocation. Archive EnTxt/XhTxt labels; count bigram LM, alpha=100.',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1))
    fig.savefig(out/'phase5_language_allocation_bpc.png',dpi=185)
    plt.close(fig)
    t=pd.read_csv(out/'per_model_view_metrics.csv')
    a=t[t.view=='en'].groupby(['family','period','arm']).bpc.mean().reset_index()
    fig,ax=plt.subplots(figsize=(9,5.4))
    for family in ('bpe','unigram'):
        for period in ('1888','1889-94'):
            d=a[(a.family==family)&(a.period==period)].set_index('arm')
            ref=d.loc['PP','bpc'];x=['PP','PW','WP','WW'];y=[d.loc[k,'bpc']-ref for k in x]
            ax.plot(x,y,marker='o' if family=='bpe' else 's',linewidth=2,label=f'{family.upper()}, {period}')
    ax.axhline(0,linestyle='--',linewidth=1,color='.4')
    ax.set_ylabel('English-view BPC difference versus PP (lower = better)')
    ax.set_xlabel('Sampling arm (A: language allocation; B: within-view selection)')
    ax.set_title('Isolating language allocation reveals an effect\nthat a simple PP-to-WW comparison can conceal')
    ax.grid(axis='y',alpha=.25)
    ax.legend(ncol=2,frameon=False)
    fig.text(.02,.02,'PP: page/page   PW: page/word   WP: word/page   WW: word/word. Count bigram LM, alpha=100.',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1))
    fig.savefig(out/'phase5_policy_offset.png',dpi=185)
    plt.close(fig)

if __name__=='__main__':main()
