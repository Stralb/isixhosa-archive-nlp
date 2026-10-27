#!/usr/bin/env python3
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent
r=json.loads((D/'RESULTS.json').read_text()); p=json.loads((D/'LINE_PLACEBO_RESULTS.json').read_text())
seed=np.arange(3);actual=np.array([z['actual_S'] for z in r['H30C']['per_seed']]);word=np.array([z['random_source_S'] for z in r['H30C']['per_seed']]);line=np.array([z['line_preserving_placebo_sum'] for z in p['per_seed']])
fig,ax=plt.subplots(figsize=(8.6,4.8));width=.24
ax.bar(seed-width,actual,width,label='Source-specific training')
ax.bar(seed,word,width,label='Random word-level control')
ax.bar(seed+width,line,width,label='Random passage-level control')
ax.axhline(0,linewidth=.8,color='black',alpha=.5)
ax.set_xticks(seed,['Seed 0','Seed 1','Seed 2']);ax.set_ylabel('Combined specialist advantage\n(pieces saved per 100 letters)')
ax.set_title('Newspaper-specific BPE benefits exceed source-blind controls',loc='left',pad=12)
ax.legend(loc='upper right',frameon=False,ncol=1);ax.grid(axis='y',alpha=.15);ax.spines[['top','right']].set_visible(False)
fig.tight_layout();fig.savefig(D/'phase30_actual_vs_controls.png',dpi=170);plt.close(fig)

q=[z for z in csv.DictReader((D/'shared_word_frequency_vs_pieces.csv').open()) if z['seed']=='0']
x=np.array([float(z['train_frequency_log_ratio']) for z in q]);y=np.array([int(z['isig_minus_imvo_pieces']) for z in q]);rng=np.random.default_rng(30246)
fig,ax=plt.subplots(figsize=(8.4,5));ax.scatter(x,y+rng.normal(0,.055,len(y)),s=15,alpha=.24,linewidths=0)
ax.axhline(0,color='black',alpha=.3);ax.axvline(0,color='black',alpha=.3)
ax.set(xlabel='Log frequency ratio: Imvo / Isigidimi (1885+1887)',ylabel='BPE pieces: Isigidimi model - Imvo model');ax.set_title('Identical held-out words, different learned segmentations',loc='left',pad=13)
ax.text(.02,.97,f'Seed 0   n={len(q)} types   Spearman rho={r["H30A"][0]["all"]["rho"]:.3f}',transform=ax.transAxes,va='top',fontsize=10)
ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.10)
fig.tight_layout();fig.savefig(D/'phase30_same_word_frequency.png',dpi=170);plt.close(fig)
