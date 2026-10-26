#!/usr/bin/env python3
"""Single-panel retrospective external transfer plot from archived raw results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parent
res=json.loads((root/'external_retro_results.json').read_text())['datasets']
keys=['ityala','zem','modern_vukuzenzele']; labels=['Ityala Lamawele','Zemk\'inkomo','Vuk\'uzenzele']
policies=['first','balanced','selected','large_single']
plabels=['One sampled BPE','50/50 BPE','Best of 3 (train-only)','One 3x-size sampled BPE']
fig,ax=plt.subplots(figsize=(10.0,5.4))
x=np.arange(len(keys));width=.19
for j,(policy,lbl) in enumerate(zip(policies,plabels)):
    values=[]
    for k in keys:
        d=res[k]['per_policy'];values.append(d[policy]['unseen_mean']-d['full_pool']['unseen_mean'])
    pos=x+(j-1.5)*width
    bars=ax.bar(pos,values,width=.18,label=lbl)
    for rect,val in zip(bars,values):
        ax.text(rect.get_x()+rect.get_width()/2,val+.025,f'{val:.2f}',ha='center',va='bottom',fontsize=8)
ax.axhline(0,linestyle='--',linewidth=.8)
ax.set_xticks(x,labels)
ax.set_ylabel('Excess pieces / 100 chars vs full-Imvo-pool BPE')
ax.set_title('Retrospective external unseen-form representation cost')
ax.set_ylim(0,1.9)
ax.legend(loc='upper left',fontsize=9,frameon=False)
fig.text(.12,.02,'All policies frozen; corpora previously inspected. Lower bars are better. Full-pool BPE is the zero reference.',fontsize=8)
fig.tight_layout(rect=(0,.05,1,1))
fig.savefig(root/'phase8_external_policy_excess.png',dpi=170)
plt.close(fig)
print('Wrote',root/'phase8_external_policy_excess.png')
