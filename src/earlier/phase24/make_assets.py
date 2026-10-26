"""Summary images; source-macro numeric data from verified result file only."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent
r=json.loads((O/'phase24_results.json').read_text())
keys=('cross_issue_repeat','bm25_evidence_minus_origin','model_comparison_diff_in_diff')
labels=['Cross-issue exact copies','BM25 strict-label undercount','Model-comparison change']
fig,ax=plt.subplots(figsize=(10.7,5.3))
x=np.arange(3);width=.33
for ind,fr in enumerate(('A','B')):
 vals=np.array([r['metrics'][fr][k]['macro']*100 for k in keys]); lo=np.array([r['metrics'][fr][k]['ci95'][0]*100 for k in keys]); hi=np.array([r['metrics'][fr][k]['ci95'][1]*100 for k in keys]);
 bars=ax.bar(x+(-.5 if fr=='A' else .5)*width,vals,width,label=('Paragraph-oriented' if fr=='A' else 'Text-position-weighted'),yerr=np.vstack([vals-lo,hi-vals]),capsize=3)
 for jj,(b,v) in enumerate(zip(bars,vals)):
  y=hi[jj]+.4 if v>=0 else lo[jj]-.35
  ax.annotate(f'{v:+.1f}' if v<0 else f'{v:.1f}',(b.get_x()+b.get_width()/2,y),ha='center',va=('bottom' if v>=0 else 'top'),fontsize=9)
ax.axhline(0,linewidth=.8,color='gray');ax.set_xticks(x);ax.set_xticklabels(labels)
ax.set_ylabel('Percentage points / percentage rate')
ax.set_title('Phase 24: sampled evidence frequency and scoring impact')
ax.legend(frameon=False,ncols=2,loc='upper right');fig.text(.5,.012,'Equal-source macro across 718 issues. Error bars: paired issue-bootstrap 95% intervals.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.025,1,1]);fig.savefig(O/'phase24_sampling_vs_scoring.png',dpi=180,bbox_inches='tight');plt.close(fig)
# Exploratory length-stratum conditional rates, clearly separated from prospective tests.
p=json.loads((O/'posthoc_sampling_diagnostics.json').read_text())
fig,ax=plt.subplots(figsize=(9.0,4.7))
lab=list(p['by_frame']['A']['repeat_by_paragraph_tokens'])
for fr in ('A','B'):
 vals=[p['by_frame'][fr]['repeat_by_paragraph_tokens'][l]['rate']*100 for l in lab]
 ax.plot(lab,vals,marker='o',label=('Paragraph-oriented' if fr=='A' else 'Text-position-weighted'))
ax.set_xlabel('Source paragraph length (word tokens)');ax.set_ylabel('Share with exact cross-issue copy (%)');ax.set_title('Exploratory only: repeats are concentrated in shorter paragraphs')
ax.legend(frameon=False);fig.tight_layout();fig.savefig(O/'phase24_posthoc_paragraph_length.png',dpi=170,bbox_inches='tight');plt.close(fig)
print('figures created')
