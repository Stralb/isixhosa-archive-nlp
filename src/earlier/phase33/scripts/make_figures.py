#!/usr/bin/env python3
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np,pandas as pd,os
r=Path(os.environ.get('PHASE33_ROOT','/mnt/data/isiXhosa_phase33_sampling_to_models'));o=r/'outputs'
tr=pd.read_csv(o/'training_audit.csv'); pos=pd.read_csv(r/'POSTHOC_PAGE_MECHANISM.csv')
pol=['P','M','W']; names=['Uniform pages','Share-matched','Word proportional']
s=tr.groupby('policy').training_en_alpha_share.agg(['mean','std']).loc[pol]*100
fig,ax=plt.subplots(figsize=(8.5,4.8));x=np.arange(3)
ax.bar(x,s['mean'].values,yerr=s['std'].values,capsize=4)
ax.set_xticks(x,names);ax.set_ylabel('Training characters from English-view text (%)');ax.set_title('Same archival text, different training exposure')
ax.set_ylim(0,35)
for i,v in enumerate(s['mean']):ax.text(i,v+1.4,f'{v:.1f}%',ha='center',fontweight='bold')
fig.tight_layout();fig.savefig(r/'figure_sampling_policy_exposure.png',dpi=170);plt.close(fig)

p=pd.read_csv(r/'POSTHOC_YEARLY_CONTRASTS.csv');p['cohort']=np.where(p.year==1888,'1888','1889-1894')
# recompute pooled from scores instead of averaging year effects
b=pd.read_csv(o/'paired_contrasts.csv');u=pd.read_csv(r/'unigram/paired_contrasts.csv')
rows=[]
for fam,data in [('BPE',b),('Unigram',u)]:
 for cohort in ('heldout_1888','later_1889_1894'):
  for v in ('en','xh'):
   q=data[(data.cohort==cohort)&(data.view==v)&(data.contrast=='P-W')&(data.metric=='pieces')&(data.resample_unit=='date')].iloc[0]
   rows.append((fam,cohort,v,q.estimate,q.interval_low,q.interval_high))
df=pd.DataFrame(rows,columns=['family','cohort','view','value','lo','hi'])
labels=['1888 English view','1888 isiXhosa view','1889–94 English view','1889–94 isiXhosa view']
fig,ax=plt.subplots(figsize=(9.8,5.4));xx=np.arange(4);w=.36
for j,fam in enumerate(['BPE','Unigram']):
 q=df[df.family==fam].sort_values(['cohort','view'],ascending=[False,True])
 # explicit row order to avoid unstable lexical sort
 vals=[];lows=[];ups=[]
 for coh,v in [('heldout_1888','en'),('heldout_1888','xh'),('later_1889_1894','en'),('later_1889_1894','xh')]:
  t=df[(df.family==fam)&(df.cohort==coh)&(df.view==v)].iloc[0]
  vals.append(t.value);lows.append(t.value-t.lo);ups.append(t.hi-t.value)
 ax.bar(xx+(j-.5)*w,vals,w,label=fam)
 ax.errorbar(xx+(j-.5)*w,vals,yerr=[lows,ups],fmt='none',capsize=3,color='black',linewidth=.9)
ax.set_xticks(xx,labels);ax.tick_params(axis='x',labelsize=9)
ax.axhline(0,color='black',linewidth=.8);ax.set_ylabel('Page minus word tokenizer cost (pieces / 100 letters)')
ax.set_title('Sampling redistributes representation efficiency; two tokenizer families')
ax.legend(frameon=False);fig.tight_layout();fig.savefig(r/'figure_sampling_tokenizer_tradeoff.png',dpi=170);plt.close(fig)

l=pd.read_csv(o/'paired_contrasts.csv');fig,ax=plt.subplots(figsize=(8.5,5))
for cohort,label in [('heldout_1888','1888'),('later_1889_1894','1889–1894')]:
 y=[];lo=[];hi=[]
 for alpha in (1,10,100):
  t=l[(l.cohort==cohort)&(l.view=='en')&(l.contrast=='P-W')&(l.metric==f'lm_bits_{alpha}')&(l.resample_unit=='date')].iloc[0]
  y.append(t.estimate);lo.append(t.estimate-t.interval_low);hi.append(t.interval_high-t.estimate)
 ax.errorbar([1,10,100],y,yerr=[lo,hi],marker='o',capsize=4,label=label)
ax.set_xscale('log');ax.axhline(0,color='black',linewidth=.8)
ax.set_xticks([1,10,100],['1','10','100']);ax.set_xlabel('Frozen unigram-backoff smoothing strength (α)')
ax.set_ylabel('Page minus word English-view loss (bits/letter)')
ax.set_title('Downstream LM conclusion is smoothing-sensitive')
ax.legend();fig.tight_layout();fig.savefig(r/'figure_downstream_smoothing_sensitivity.png',dpi=170);plt.close(fig)

fig,ax=plt.subplots(figsize=(7.4,4.7));bar=pos.contribution_W_minus_P.to_numpy()*100
ax.bar(pos.page.astype(str),bar);ax.axhline(0,color='black',linewidth=.8)
ax.set_ylabel('Contribution to English-view word exposure difference (pp)')
ax.set_xlabel('Historical newspaper page position')
ax.set_title('Page 3 drives the sampling difference (1885 and 1887)')
for i,n in enumerate(bar):ax.text(i,n+(.25 if n>=0 else -.5),f'{n:+.2f}',ha='center')
fig.tight_layout();fig.savefig(r/'figure_page_position_mechanism.png',dpi=170);plt.close(fig)
print('FIGURES GENERATED')
