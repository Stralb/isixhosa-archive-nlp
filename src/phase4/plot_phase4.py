#!/usr/bin/env python3
"""Reproduce Phase 4 figures from complete saved score contrasts."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
 out=Path(args.output)
 tab=pd.read_csv(out/'f6_contrasts.csv');original=pd.read_csv(out/'phase3_original_reference.csv')
 x=np.arange(2);periods=['1888','1889-94']
 for view in ('en','xh'):
  fig,ax=plt.subplots(figsize=(8,4.65),layout='constrained')
  records=[('Earlier Phase 3 (all eligible pages)','original_PP_minus_WW','o'),
   ('Same-support, unfiltered','support_control_PP_minus_WW','s'),
   ('Same-support, duplicate-filtered','filtered_PP_minus_WW','^')]
  for j,(label,key,marker) in enumerate(records):
   xs=x+(j-1)*.17
   if j==0:
    r=original[original.view==view].set_index('period').loc[periods]
    values=r.original_PP_minus_WW.to_numpy(float)
    low=r.original_ci_low.to_numpy(float);high=r.original_ci_high.to_numpy(float)
   else:
    r=tab[(tab.view==view)&(tab.contrast==key)].set_index('period').loc[periods]
    values=r['mean'].to_numpy(float);low=r.ci_low.to_numpy(float);high=r.ci_high.to_numpy(float)
   ax.errorbar(xs,values,yerr=[values-low,high-values],fmt=marker,capsize=4,markersize=7,label=label)
  ax.axhline(0,linewidth=1,linestyle='--',alpha=.7)
  ax.set_xticks(x,periods)
  ax.set_ylabel('PP minus WW (pieces per 100 alphabetic characters)')
  ax.set_title('Repeated-text exclusion: '+('English-view' if view=='en' else 'isiXhosa-view')+' tokenization')
  ax.legend(fontsize=8,loc='best')
  fig.savefig(out/f'phase4_{view}_repetition_robustness.png',dpi=170)
  plt.close(fig)
if __name__=='__main__':main()
