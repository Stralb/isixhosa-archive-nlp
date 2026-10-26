#!/usr/bin/env python3
"""Render two figures from VERIFIED existing P3/P5 contrasts; no new science."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_pair(tab, contrast, view, title, unit, fname, xlim=None):
    fig,axs=plt.subplots(1,2,figsize=(10.4,3.7),sharey=True,layout='constrained')
    rowtypes=[('bpe','BPE'),('unigram','Unigram')]
    yticks=np.array([0,1,2,3])
    for ax,period in zip(axs,['1888','1889-94']):
        subset=tab[(tab.period==period)&(tab.contrast==contrast)&(tab.view==view)]
        for fam,label in rowtypes:
            rec=subset[subset.family==fam].iloc[0]
            yi=1 if fam=='bpe' else 0
            ax.errorbar(float(rec['mean']),yi,xerr=[[float(rec['mean']-rec.ci_low)],[float(rec.ci_high-rec['mean'])]],fmt='o',capsize=4,markersize=7,label=label)
            ax.annotate(f"{rec['mean']:+.4f}" if 'bits / character' in unit else f"{rec['mean']:+.3f}", (float(rec['mean']), yi),xytext=(0,12),textcoords='offset points',ha='center',fontsize=10)
        ax.axvline(0,color='0.5',lw=1,linestyle='--')
        ax.set_yticks([0,1],['Unigram','BPE'])
        ax.set_ylim(-0.6,1.6)
        ax.grid(axis='x',color='0.88')
        ax.set_xlabel(unit)
        ax.set_title('1888 held-out' if period=='1888' else '1889–1894 out-of-period', fontsize=11)
        if xlim is not None:ax.set_xlim(*xlim)
    fig.suptitle(title,fontsize=12,weight='bold')
    fig.savefig(fname,dpi=210,bbox_inches='tight')
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    r=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    t=pd.read_csv(r/'Phase3/outputs/factorial_contrasts.csv')
    b=pd.read_csv(r/'Phase5/outputs/bpc_factorial_contrasts.csv')
    plot_pair(t,'allocation','en','A. English-view segmentation: isolated allocation factor','Change in pieces / 100 letters (lower is better)',out/'PHASE6_ENGLISH_SEGMENTATION.png')
    plot_pair(b,'allocation','en','B. English-view bigram prediction: isolated allocation factor','Change in bits / character (lower is better)',out/'PHASE6_ENGLISH_LM.png')
    plot_pair(b,'WW_minus_PP','en','C. Full sampling-policy contrast: WW minus PP','Change in bits / character (lower is better)',out/'PHASE6_WHOLE_POLICY_COUNTEREXAMPLE.png')
    plot_pair(b,'allocation','xh','D. isiXhosa-view bigram prediction: isolated allocation','Change in bits / character (lower is better)',out/'PHASE6_XH_TRADEOFF.png')
    print('Four figures generated with existing 95% publication-issue bootstrap intervals.')
if __name__=='__main__':main()
