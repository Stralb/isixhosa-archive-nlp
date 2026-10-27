import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent
j=json.loads((ROOT/'PHASE11_RESULTS.json').read_text()); t=json.loads((ROOT/'temporal'/'TEMPORAL_RESULTS.json').read_text())
fig,ax=plt.subplots(figsize=(8.5,5.2))
for source, dat in j['summary'].items():
 x=[100*r['relative_margin'] for r in dat['rows']]
 y=[r['scores']['greedy_regret_E'] for r in dat['rows']]
 ax.scatter(x,y,label=source,s=48,alpha=.85)
ax.axvline(2,color='0.4',linestyle='--',lw=1.2,label='Frozen 2% warning')
ax.axhline(.25,color='0.55',linestyle=':',lw=1.2,label='Meaningful regret')
ax.set(xlabel='Initial first-versus-second pair frequency margin (%)',
       ylabel='Greedy first-merge regret on unseen forms\n(pieces / 100 alphabetic characters)',
       title='A near-tie warns of possible failure, not inevitable regret')
ax.legend(loc='upper right',fontsize=9)
ax.grid(alpha=.17)
fig.tight_layout()
fig.savefig(ROOT/'phase11_margin_vs_regret.png',dpi=175)
plt.close(fig)
fig,ax=plt.subplots(figsize=(8.5,4.8))
import collections
cohorts=[('Imvo 1887',j['summary']['IMVO']['rows'],'scores','greedy_regret_E'),
         ('Imvo 1889',t['results']['1889']['rows'],'outcome','regret'),
         ('Imvo 1890',t['results']['1890']['rows'],'outcome','regret')]
for idx,(label,rs,k,k2) in enumerate(cohorts):
 ys=np.array([r[k][k2] for r in rs]);xx=np.full(len(ys),idx)+np.linspace(-.16,.16,len(ys))
 ax.scatter(xx,ys,s=44,alpha=.9,label=label)
 ax.plot([idx-.25,idx+.25],[ys.mean(),ys.mean()],lw=2.6,color='black')
ax.axhline(.25,color='0.55',linestyle=':',label='0.25 E meaningful threshold')
ax.set_xticks([0,1,2],[x[0] for x in cohorts]);ax.set_xlim(-.5,2.5)
ax.set(ylabel='Greedy first-merge regret (pieces / 100 characters)',
       title='The Imvo failure is not stable across historical periods')
ax.grid(axis='y',alpha=.18)
fig.tight_layout();fig.savefig(ROOT/'phase11_temporal_regret.png',dpi=175)
plt.close(fig)
