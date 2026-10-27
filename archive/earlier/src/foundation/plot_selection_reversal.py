#!/usr/bin/env python3
"""Regenerate diagnostic figure from recorded experimental outputs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/mnt/data/isiXhosa_publication_continuation')
R=json.loads((ROOT/'mask_sensitivity_results.json').read_text())
fig,ax=plt.subplots(figsize=(7.2,4.4))
for k,style in [('validation','o-'),('old_test','s-')]:
    xx=[];yy=[];lower=[];upper=[]
    for n in (8,12,20):
        rec=R['ns'][str(n)][k]
        v=rec['delta_macro_nonmatch_BPC']; ci=rec['bootstrap_95_CI']['macro_nonmatch']['ci_95']
        xx.append(n);yy.append(v);lower.append(v-ci[0]);upper.append(ci[1]-v)
    ax.errorbar(xx,yy,yerr=[lower,upper],fmt=style,label=('1887 validation' if k=='validation' else '1888 retrospective'),capsize=4)
ax.axhline(0,linestyle='--',linewidth=1)
ax.set_xticks([8,12,20])
ax.set_xlabel('Exact matching passage length (words)')
ax.set_ylabel('BPC, balanced minus hard dedup (nonmatching text)')
ax.set_title('Selection reversal survives overlap-mask sensitivity')
ax.legend(frameon=False)
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(ROOT/'selection_reversal_sensitivity.png',dpi=180)
plt.close(fig)

r=json.loads((ROOT/'selection_reversal_results.json').read_text())
fig,ax=plt.subplots(figsize=(7.2,4.4))
for k,label in [('validation','1887 validation'),('old_test','1888 retrospective')]:
    x=r['splits'][k]['summary']; a=x['BALANCED']['domains']['IMVO']; b=x['HARD']['domains']['IMVO']
    new=a['nonmatch']-b['nonmatch'];rep=a['match']-b['match']; actual=a['chars_match']/a['chars'];root=new/(new-rep)
    p=[x/1000 for x in range(0,251)]
    yy=[(1-z)*new+z*rep for z in p]
    ax.plot([v*100 for v in p],yy,label=f'{label}: crossover {root*100:.2f}%')
    ax.scatter([actual*100],[(1-actual)*new+actual*rep],s=65)
ax.axhline(0,linestyle='--',linewidth=1)
ax.set_xlabel('Share of evaluation text matching training passages (%)')
ax.set_ylabel('Imvo BPC difference, balanced minus hard dedup')
ax.set_title('Observed ranking depends on repetition prevalence')
ax.legend(frameon=False)
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(ROOT/'repeat_prevalence_crossover.png',dpi=180)
plt.close(fig)
print('generated figures')
