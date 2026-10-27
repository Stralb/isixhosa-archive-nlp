#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path('/mnt/data/isiXhosa_publication_continuation/phase5')
res=json.loads((ROOT/'results.json').read_text())
policies=['RAW','BALANCED','HARD','SOFT','SOFT_BALANCED','PERSIST_SOFT','TRANSIENT_SOFT']
years=('1893','1894')
fig,ax=plt.subplots(figsize=(10.4,4.8))
x=np.arange(len(policies))
for i,year in enumerate(years):
    means={p:np.mean([record['heldout'][year]['nonmatch'] for record in res['by_policy'][p].values()]) for p in policies}
    deltas=[1000*(means[p]-means['RAW']) for p in policies]
    ax.plot(x,deltas,marker='o',linewidth=2,label=year)
ax.axhline(0,ls='--',lw=1,color='gray')
ax.set_xticks(x,labels=['Raw','Balanced','Hard dedup','Soft all','Soft + balance','Persistent only','Transient only'],rotation=12,ha='right')
ax.set_ylabel('BPC difference from raw (x 1,000)')
ax.set_title('Previously unused 1893–1894 Imvo issues: held-out nonmatching text')
ax.legend(title='Evaluation year')
ax.grid(axis='y',alpha=.18)
fig.tight_layout()
fig.savefig(ROOT/'phase5_policy_comparison.png',dpi=180)
print(ROOT/'phase5_policy_comparison.png')
