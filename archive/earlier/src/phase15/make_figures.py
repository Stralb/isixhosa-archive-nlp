import json,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
O=Path('/mnt/data/isiXhosa_phase15_source_grounded_search')
S=json.load(open(O/'paragraph_benchmark_summary.json'));R=json.load(open(O/'paragraph_retrieval_summary.json'));A=json.load(open(O/'archive_language_access_mass.json'))
labels=['Imvo Xh','Imvo En','Isigidimi Xh','Isigidimi En'];keys=['IMVO_xh','IMVO_en','ISIG_xh','ISIG_en']
fig,ax=plt.subplots(figsize=(9,4.5))
x=np.arange(4)
segment=[100*(S[k]['not_verbatim_paragraph']-S[k]['not_verbatim_flat'])/S[k]['view_windows'] for k in keys]
flat=[100*S[k]['not_verbatim_flat_frac'] for k in keys]
ax.bar(x-.18,flat,width=.35,label='Absent from flattened raw page')
ax.bar(x+.18,[100*S[k]['not_verbatim_paragraph_frac'] for k in keys],width=.35,label='Absent from any raw paragraph')
ax.set_xticks(x,labels);ax.set_ylabel('Percent of 10-word windows');ax.set_title('Boundary handling changes apparent text provenance');ax.legend(loc='upper left',fontsize=9);ax.grid(axis='y',alpha=.15)
ax.text(.01,-.20,'These are data-processing artifacts, not verified linguistic errors.',transform=ax.transAxes,fontsize=9)
fig.tight_layout();fig.savefig(O/'figure_boundary_integrity.png',dpi=180,bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4.5))
methods=['raw','fielded_auto','fielded_oracle'];names=['Raw-page BM25','Automatic bilingual fields','Oracle bilingual fields']
x=np.arange(len(methods));width=.36
clean=[R[m]['clean_all']['any1']*100 for m in methods];noisy=[R[m]['noisy_all']['any1']*100 for m in methods]
ax.bar(x-width/2,clean,width,label='Clean paragraph-valid queries');ax.bar(x+width/2,noisy,width,label='Synthetic OCR-noisy queries')
ax.set_xticks(x,names);ax.set_ylim(0,105);ax.set_ylabel('Any valid-source page at rank 1 (%)');ax.set_title('Evidence-aware retrieval: 583 queries / 3,608 pages');ax.legend(loc='lower left',fontsize=9);ax.grid(axis='y',alpha=.15)
for xi,v in enumerate(noisy):ax.text(xi+width/2,v+1,f'{v:.1f}%',ha='center',fontsize=9)
fig.tight_layout();fig.savefig(O/'figure_retrieval_results.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,3.6))
keys=['IMVO','ISIG'];ret=[A[k]['english_majority_english_retained_frac']*100 for k in keys];lost=[100-v for v in ret]
ax.barh(keys,ret,label='English words retained (English-majority pages)');ax.barh(keys,lost,left=ret,label='English words hidden by the hypothetical filter')
for i,v in enumerate(ret):ax.text(v+1,i,f'{lost[i]:.1f}% hidden',va='center',fontsize=10)
ax.set_xlim(0,110);ax.set_xlabel('Share of English-language text in source (%)');ax.set_title('Hypothetical majority-language page filter');ax.legend(loc='lower right',fontsize=8);ax.invert_yaxis();fig.tight_layout();fig.savefig(O/'figure_majority_filter_word_mass.png',dpi=180);plt.close(fig)
print('FIGURES',*[p.name for p in O.glob('figure_*.png')])
