#!/usr/bin/env python3
"""Phase 29 result figures from frozen JSON files. No secondary hypotheses."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
D=Path(__file__).parent
A=json.loads((D/'RESULTS.json').read_text());B=json.loads((D/'ORDER_SENSITIVITY_RESULTS.json').read_text());C=json.loads((D/'LENGTH_STANDARDIZATION_RESULTS.json').read_text())
labels=['Original\nchar_wb','Word\nunigram','Word\n1–2grams','Cross-word\nchar n-grams','Word-length\nsequence','Length-bin\nhistogram','No exact\nreprints']
vals=[100*A['conditions']['ordered']['macro'],100*B['ordered']['word1'],100*B['ordered']['word12'],100*B['ordered']['cross_char'],100*A['conditions']['shape_ordered']['macro'],100*A['conditions']['shape_histogram']['macro'],100*A['conditions']['no_repeat_ordered']['macro']]
fig,ax=plt.subplots(figsize=(11,5.2));x=np.arange(len(labels));ax.bar(x,vals,alpha=.85);ax.axhline(50,ls='--',linewidth=1,label='Balanced chance: 50%')
for i,v in enumerate(vals):ax.annotate(f'{v:.1f}%',(i,v),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9)
ax.set_ylim(40,86);ax.set_xticks(x);ax.set_xticklabels(labels,fontsize=9);ax.set_ylabel('Issue-balanced source accuracy (%)');ax.set_title('Phase 29 · Source prediction without evidence of historical meaning');ax.legend(loc='lower right',frameon=False);ax.spines[['top','right']].set_visible(False)
fig.tight_layout();fig.savefig(D/'phase29_source_signatures.png',dpi=180);plt.close(fig)

fig,ax=plt.subplots(figsize=(7.5,5.2));sources=['Imvo','Isigidimi'];raw=[C['means'][s]['raw'] for s in ('IMVO','ISIG')];std=[C['means'][s]['standardized'] for s in ('IMVO','ISIG')];xx=np.arange(2);bar=.32
ax.bar(xx-bar/2,raw,width=bar,label='Original');ax.bar(xx+bar/2,std,width=bar,label='Standardized to common word-length mix')
for i in range(2):
 ax.annotate(f'{raw[i]:.3f}',(i-bar/2,raw[i]),xytext=(0,5),textcoords='offset points',ha='center')
 ax.annotate(f'{std[i]:.3f}',(i+bar/2,std[i]),xytext=(0,5),textcoords='offset points',ha='center')
ax.set_xticks(xx);ax.set_xticklabels(sources);ax.set_ylim(0,1.15);ax.set_ylabel('Specialist advantage (pieces saved / 100 alphabetic chars)');ax.set_title('Phase 29 · Equalizing word-length composition changes little');ax.spines[['top','right']].set_visible(False);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.13),frameon=False)
fig.tight_layout();fig.savefig(D/'phase29_length_standardization.png',dpi=180);plt.close(fig)
