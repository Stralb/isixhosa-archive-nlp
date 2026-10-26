#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('/mnt/data/isiXhosa_publication_continuation/phase3')
r=json.loads((p/'independent_imvo_results.json').read_text())
fig,ax=plt.subplots(figsize=(8.6,5.2))
x=np.linspace(0,0.105,240)
for y in ['1889','1890']:
    d=r['policy_status'][y]['BALANCED_minus_HARD']
    q=r['years'][y]['HARD']['matched_fraction']
    curve=(1-x)*d['nonmatch']+x*d['match']
    line,=ax.plot(x*100,curve,label=f'{y}: fixed model pair')
    ax.plot(q*100,d['all'],marker='o',markersize=8,color=line.get_color())
    cross=d['nonmatch']/(d['nonmatch']-d['match'])
    ax.text(cross*100+0.15,.008 if y=='1889' else -.015,
            f'{y} crossover {100*cross:.2f}%',fontsize=9,color=line.get_color())
ax.axhline(0,linewidth=1,color='black',alpha=.7)
ax.set_xlabel('Fraction of characters in train-matching passages (%)')
ax.set_ylabel('Balanced minus hard-dedup BPC (lower is better)')
ax.set_title('Exact reuse share changes the preferred model on full text')
ax.legend(frameon=False)
ax.grid(alpha=.18)
ax.text(.01,.02,'Markers = observed annual repeat prevalence',transform=ax.transAxes,fontsize=9)
fig.tight_layout()
fig.savefig(p/'phase3_crossover_independent.png',dpi=190)
plt.close(fig)

fig,ax=plt.subplots(figsize=(8.6,5.1))
policies=['RAW','BALANCED','HARD','SOFT','SOFT_BALANCED']
xt=np.arange(len(policies))
y1889=[r['years']['1889'][k]['nonmatch'] for k in policies]
y1890=[r['years']['1890'][k]['nonmatch'] for k in policies]
ax.plot(xt,y1889,'o-',label='1889 (50 issues)')
ax.plot(xt,y1890,'s-',label='1890 (52 issues)')
ax.set_xticks(xt)
ax.set_xticklabels(['Raw','Balanced','Hard\ndedup','Soft','Soft +\nbalanced'],fontsize=9)
ax.set_ylabel('Nonmatching bits per character (lower is better)')
ax.set_title('Previously unused Imvo evaluation years: all five fixed policies')
ax.grid(axis='y',alpha=.2)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(p/'phase3_policy_independent.png',dpi=190)
plt.close(fig)
print('saved figures')
