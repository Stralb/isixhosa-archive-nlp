"""Plot only frozen Phase 20 statistics; never modify evaluation."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
P=Path(__file__).resolve().parent
R=json.loads((P/'phase20_results.json').read_text())
forms=['keyword','clean','synthetic_noise'];fl=['Spaced source keywords','Clean 5-word excerpt','Synthetic-noise excerpt']
methods=[('raw_bm25','Plain BM25'),('raw_fullfuzzy','Full-page fuzzy'),('gated_fullfuzzy','Gated + fuzzy'),('raw_local120','Local lexical (120 words)')]
fig,ax=plt.subplots(figsize=(10.5,5.8));x=np.arange(3)
for method,label in methods:
 y=[100*R['method_metrics'][f+'_'+method]['anchor_top1']['macro'] for f in forms]
 ci=[R['method_metrics'][f+'_'+method]['anchor_top1']['ci95'] for f in forms]
 lo=[y[i]-ci[i][0]*100 for i in range(3)];hi=[ci[i][1]*100-y[i] for i in range(3)]
 ax.errorbar(x,y,yerr=[lo,hi],marker='o',capsize=3,linewidth=1.7,label=label)
ax.set_xticks(x,fl);ax.set_ylim(0,110);ax.set_ylabel('Top-1 originating-page recovery (%)')
ax.grid(axis='y',alpha=.25)
ax.set_title('Phase 20: the same evidence pages, three query construction regimes',loc='left',fontsize=12)
ax.legend(loc='lower left',ncol=2,framealpha=.95)
fig.text(.09,.01,'Issue-bootstrap 95% intervals; macro-averaged across 4 archive source/view cells (n = 50, 4, 50, 50).\nSource-derived proxies and synthetic corruption, not independent human queries or authentic OCR.',fontsize=8)
fig.tight_layout(rect=[0,.08,1,1]);fig.savefig(P/'phase20_query_form_effect.png',dpi=180);plt.close(fig)

fig,ax=plt.subplots(figsize=(8.9,4.9))
for i,key in enumerate(('H20a_interaction','H20b_keyword_local_minus_bm25')):
 t=R['comparisons'][key];y=100*t['macro'];lo,hi=[100*v for v in t['ci95']]
 ax.errorbar([y],[i],xerr=[[y-lo],[hi-y]],fmt='o',capsize=4)
 ax.text(y+2,i+0.13,f'{y:+.1f} pp [{lo:+.1f}, {hi:+.1f}]',fontsize=10)
ax.axvline(0,linestyle=':',color='gray');ax.set_xlim(-8,104);ax.set_ylim(-.6,1.7)
ax.set_yticks([0,1],['H20a: fuzzy interaction\n(noisy vs keywords)','H20b: local reranker\n(keywords vs BM25)'])
ax.set_xlabel('Paired Top-1 difference (percentage points)')
ax.set_title('Preregistered gates: query-form interaction supported; reranker transfer rejected',loc='left',fontsize=11)
fig.tight_layout();fig.savefig(P/'phase20_hypothesis_tests.png',dpi=180);plt.close(fig)
