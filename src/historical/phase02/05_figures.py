#!/usr/bin/env python3
"""Produce deterministic benchmark diagnostic figures from frozen summary CSVs."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABEL={'MAR_SOURCE_PAGE':'Source + page MAR','MAR_LENGTH_DATE':'Length/date MAR','MNAR_LATENT':'Hidden-text MNAR','ISSUE_DROPOUT':'Whole-issue loss','ZERO_CELL':'Zero-support cell','REPEAT_PREF':'Repeated-text preference'}

def plot(root):
 root=Path(root); dev=pd.read_csv(root/'outputs/dev/summary_metrics.csv',keep_default_na=False)
 test=pd.read_csv(root/'outputs/test/summary_metrics.csv',keep_default_na=False)
 out=root/'figures';out.mkdir(parents=True,exist_ok=True)
 for df in (dev,test):
  for c in ['mae_per_10000','estimable_fraction','decision_coverage','null_false_sign_fraction','signal_recovery','interval_coverage','interval_width_per_10000','premise_holds']:
   df[c]=pd.to_numeric(df[c],errors='coerce')
 plt.rcParams.update({'font.size':10,'figure.dpi':140,'savefig.dpi':170,'font.family':'DejaVu Sans'})
 # Main inferential frontier by assumptions in injected world.
 z=test[(test.world=='INJECTED')&(test.kind=='set')]
 fig,ax=plt.subplots(figsize=(8,5.4))
 markers={'MANSKI':'o','GAMMA_0005':'^','GAMMA_0020':'s'}
 for method in ['MANSKI','GAMMA_0020','GAMMA_0005']:
  ds=z[z.method==method]
  ax.scatter(ds.decision_coverage*100,ds.interval_coverage*100,s=62,marker=markers[method],label={'MANSKI':'Worst-case identified set','GAMMA_0020':'Sensitivity γ = 0.0020','GAMMA_0005':'Sensitivity γ = 0.0005'}[method])
  if method=='GAMMA_0005':
   for r in ds.itertuples():
    if r.scenario not in ('MNAR_LATENT','ZERO_CELL'): continue
    ax.annotate(LABEL[r.scenario],(r.decision_coverage*100,r.interval_coverage*100),xytext=(6,-2),textcoords='offset points',fontsize=9)
 ax.set_xlim(-3,55);ax.set_ylim(45,104);ax.set_xlabel('Sign-resolved word-form decisions (%)');ax.set_ylabel('True finite-population effect inside interval (%)')
 ax.set_title('Partial identification: certainty versus useful conclusions')
 ax.axhline(95,ls=':',lw=1.0,color='gray');ax.grid(alpha=.18);ax.legend(loc='lower left',fontsize=8)
 fig.tight_layout();fig.savefig(out/'phase2_identification_tradeoff.png');plt.close(fig)
 # Breakdown of unknown latent MNAR: actual point error.
 x=test[(test.world=='INJECTED')&(test.scenario=='MNAR_LATENT')&(test.kind=='point')].copy()
 fig,ax=plt.subplots(figsize=(8,4.5));labels=['Observed mean','Source + page','Exact cell matching','Known π (oracle)','Fitted π from frame']
 inds=np.arange(len(x));bars=ax.barh(inds,x.mae_per_10000)
 ax.set_yticks(inds,labels);ax.invert_yaxis();ax.set_xlim(0,max(x.mae_per_10000)*1.22)
 ax.set_xlabel('Mean absolute effect error (occurrences per 10k page-normalized tokens)')
 ax.set_title('Outcome-dependent removal: point correction remains imperfect')
 for i,r in enumerate(x.itertuples()):ax.text(r.mae_per_10000+.08,i,f'{r.mae_per_10000:.2f}',va='center',fontsize=9)
 ax.grid(axis='x',alpha=.18);fig.tight_layout();fig.savefig(out/'phase2_mnar_baseline_errors.png');plt.close(fig)
 # Positivity: proportion estimable across 6 mechanisms.
 ds=test[(test.world=='INJECTED')&(test.kind=='point')]
 methods=['POOLED','SOURCE_PAGE_STD','EXACT_CELL_MATCH','KNOWN_IPW_ORACLE','EST_IPW_FRAME']
 names=['Naive','Standardized','Matched','Oracle IPW','Fitted IPW']
 piv=ds.pivot(index='scenario',columns='method',values='estimable_fraction').reindex(LABEL)
 fig,ax=plt.subplots(figsize=(10,4.6))
 width=.15; xx=np.arange(len(piv))
 for j,method in enumerate(methods):ax.bar(xx+(j-2)*width,piv[method],width,label=names[j])
 ax.set_xticks(xx,[LABEL[i].replace(' ','\n') for i in piv.index],fontsize=8)
 ax.set_ylim(0,1.09);ax.set_ylabel('Fraction of replicates with an estimate');ax.set_title('Lack of source–page support invalidates some estimators')
 ax.legend(ncol=3,fontsize=8);ax.grid(axis='y',alpha=.15);fig.tight_layout();fig.savefig(out/'phase2_positivity_gate.png');plt.close(fig)
 # Dev/test comparison of MNAR source/page and fitted estimates + misspecification bound.
 fig,axes=plt.subplots(1,2,figsize=(10.5,4))
 m=['SOURCE_PAGE_STD','EST_IPW_FRAME','GAMMA_0005','GAMMA_0020']
 labs=['Stratification','Fitted propensity','γ=.0005 bounds','γ=.0020 bounds']
 for ax,metric,ttl in zip(axes,['null_false_sign_fraction','signal_recovery'],['False signs on known-null forms','Correct signs on injected forms']):
  xs=np.arange(len(m));wd=.34
  for i,(df,label) in enumerate([(dev,'Development'),(test,'Held-out simulation')]):
   q=df[(df.world=='INJECTED')&(df.scenario=='MNAR_LATENT')].set_index('method').loc[m]
   ax.bar(xs+(i-.5)*wd,q[metric]*100,wd,label=label)
  ax.set_xticks(xs,labs,rotation=25,ha='right',fontsize=8);ax.set_ylabel('Word-form decisions (%)');ax.set_title(ttl);ax.grid(axis='y',alpha=.15)
 axes[0].legend(fontsize=8);fig.suptitle('Issue-held-out simulation transfer: MNAR selection',fontsize=12)
 fig.tight_layout();fig.savefig(out/'phase2_dev_test_reproduction.png');plt.close(fig)
 print('FIGURES',*[str(p) for p in sorted(out.glob('*.png'))],sep='\n')

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();plot(a.root)
