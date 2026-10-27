#!/usr/bin/env python3
"""Render diagnostic figures from existing saved results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent
s=json.loads((ROOT/'boundary_sensitivity_results.json').read_text())
r=json.loads((ROOT/'masked_selection_results.json').read_text())
labels=['1887 validation\nImvo','1888 retrospective\nImvo','1888 retrospective\nIsigidimi']
indices=[('1887_validation','IMVO'),('1888_old_evaluation','IMVO'),('1888_old_evaluation','ISIG')]
original=[100*s[t][d]['page_concat_fraction'] for t,d in indices]
strict=[100*s[t][d]['paragraph_respecting_fraction'] for t,d in indices]
fig,ax=plt.subplots(figsize=(8.8,4.5),layout='constrained')
x=np.arange(len(labels));w=.34
ax.bar(x-w/2,original,w,label='Page-concatenated matching')
ax.bar(x+w/2,strict,w,label='Paragraph-respecting matching')
ax.set_xticks(x,labels)
ax.set_ylabel('1887 training match coverage (% of words)')
ax.set_title('Exact 12-word overlap is preprocessing-sensitive')
ax.legend(frameon=False)
for points,offset in ((original,-w/2),(strict,w/2)):
 for i,v in enumerate(points):ax.annotate(f'{v:.2f}%',(i+offset,v),ha='center',va='bottom',fontsize=9,xytext=(0,2),textcoords='offset points')
ax.set_ylim(0,22)
fig.savefig(ROOT/'boundary_sensitivity.png',dpi=185)
plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4.4),layout='constrained')
sp=['validation','old_test'];title=['1887 validation','1888 retrospective test']
old=[r['splits'][s]['old_delta'] for s in sp]
new=[r['splits'][s]['delta_BALANCED_minus_HARD_nonmatch'] for s in sp]
ci=[r['splits'][s]['bootstrap']['ci_95'] for s in sp]
yerr=np.array([[new[i]-ci[i][0] for i in range(2)],[ci[i][1]-new[i] for i in range(2)]])
x=np.arange(2);w=.36
ax.axhline(0,linewidth=.8,linestyle='--')
ax.bar(x-w/2,old,w,label='Page-concat overlap mask')
ax.bar(x+w/2,new,w,label='Paragraph-respecting mask',yerr=yerr,capsize=3)
ax.set_xticks(x,title);ax.set_ylabel('Balanced - hard dedup BPC on nonmatched text')
ax.set_title('Evaluation mask changes model-selection contrast')
ax.legend(frameon=False)
fig.savefig(ROOT/'masked_ranking_sensitivity.png',dpi=185)
plt.close(fig)
print('Saved figures')
