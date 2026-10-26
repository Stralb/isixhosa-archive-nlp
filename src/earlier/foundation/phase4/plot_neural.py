#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/mnt/data/isiXhosa_publication_continuation')
r3=json.loads((root/'phase3'/'independent_imvo_results.json').read_text())
r4=json.loads((root/'phase4'/'neural_results.json').read_text())
policies=['RAW','BALANCED','HARD','SOFT','SOFT_BALANCED']
labels=['Raw','Balanced','Hard dedup','Soft weighting','Soft + balance']
values=np.empty((5,4),float)
for i,p in enumerate(policies):
    values[i,0]=r3['years']['1889'][p]['nonmatch']
    values[i,1]=r3['years']['1890'][p]['nonmatch']
    for k,year in enumerate(('1891','1892'),2):
        values[i,k]=np.mean([v['test'][year]['nonmatch'] for v in r4['model_outputs'][p].values()])
ranks=np.argsort(np.argsort(values,axis=0),axis=0)+1
fig,ax=plt.subplots(figsize=(8.8,4.9))
image=ax.imshow(ranks,aspect='auto',vmin=1,vmax=5)
ax.set_xticks(range(4),['Count LM\n1889','Count LM\n1890','Neural MLP\n1891','Neural MLP\n1892'])
ax.set_yticks(range(5),labels)
for i in range(5):
    for j in range(4):
        ax.text(j,i,str(ranks[i,j]),ha='center',va='center',fontsize=12,
                color='white' if ranks[i,j]>3 else 'black',weight='bold')
ax.set_title('Policy rankings differ by model family and evaluation cohort')
ax.set_xlabel('Lower nonmatching BPC means better rank (1 is best)')
cbar=fig.colorbar(image,ax=ax)
cbar.set_label('Rank (1 best, 5 worst)')
fig.tight_layout()
out=root/'phase4'/'policy_rank_transfer.png'
fig.savefig(out,dpi=190)
print('saved',out)
