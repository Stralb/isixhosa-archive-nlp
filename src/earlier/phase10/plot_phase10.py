#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
results=json.loads((ROOT/'PHASE10_RESULTS.json').read_text())
rows={c['name']:c for c in results['contrasts']}
seed_labels=['101','202','303']
fig,axes=plt.subplots(1,2,figsize=(11.7,4.4),constrained_layout=True)
for j,what in enumerate(['newspaper_worst_bpc','ityala']):
    ax=axes[j]
    for i,(label,key) in enumerate([('Sample 5001','5001_good_greedy_vs_bad_second'),
         ('Sample 5004','5004_selected_second_vs_greedy'),('Sample 5012','5012_good_greedy_vs_bad_second')]):
        d=rows[key]['metrics'][what]
        x=np.array([i-.11,i,i+.11]); yy=np.array(d['paired_per_seed'])
        ax.scatter(x,yy,s=35,alpha=.8,label=f'Neural seed' if i==0 else None)
        ax.plot([i-.16,i+.16],[d['a_minus_b_mean']]*2,color='black',lw=2)
    ax.axhline(0,color='0.4',lw=1,linestyle='--')
    ax.set_xlim(-.5,2.5)
    ax.set_xticks([0,1,2],labels=['5001','5004','5012'])
    ax.set_ylabel('n+g minus word-start+n (BPC)\nNegative favours n+g')
    ax.set_title('Historical newspapers' if j==0 else 'Ityala Lamawele (OCR)')
    ax.grid(axis='y',alpha=.25)
fig.suptitle('The first-merge intervention helps in-domain neural prediction\nbut does not consistently transfer to historical books',fontsize=12)
fig.savefig(ROOT/'phase10_neural_transfer_split.png',dpi=200)
plt.close(fig)

fig,ax=plt.subplots(figsize=(8.3,4.4),constrained_layout=True)
keys=['imvo5001_greedy','imvo5004_second','balanced6001','fullpool_imvo']
labels=['Low-budget 5001 n+g','Low-budget 5004 n+g','Balanced BPE','Full Imvo pool BPE']
Y=np.array([[results['models'][m]['mean_bpc'][metric] for metric in ('newspaper_worst_bpc','ityala','zem','modern')] for m in keys])
# Express deviation relative to full-pool in each corpus, not absolute BPC scales.
D=(Y-Y[-1])*1000
x=np.arange(4)
for i in range(3):
    ax.plot(x,D[i],marker='o',label=labels[i])
ax.axhline(0,color='black',linestyle='--',lw=1,label=labels[-1])
ax.set_xticks(x,labels=['Newspapers\nworst-domain','Ityala','Zemk\'inkomo','Modern'])
ax.set_ylabel('Excess BPC vs full-pool BPE (×1,000)\nLower is better')
ax.set_title('Full-pool remains a strong baseline across genres')
ax.legend(loc='upper left',fontsize=8)
ax.grid(axis='y',alpha=.25)
fig.savefig(ROOT/'phase10_fullpool_baseline.png',dpi=200)
print('Wrote two figures')
