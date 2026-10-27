"""Make a deterministic Phase23 visualization from saved computed results."""
from pathlib import Path
import json,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).parent
r=json.loads((O/'phase23_results.json').read_text());m=r['metrics']['repeated']['keyword6']
methods=['BM25','BM25_FUZZY20','XH_BM25_FUZZY20'];names=['Plain BM25','BM25 + fuzzy','Xh-field + fuzzy']
strict=[100*m[k]['origin_top1']['macro'] for k in methods]
exact=[100*m[k]['equiv_top1']['macro'] for k in methods]
fig,ax=plt.subplots(figsize=(9.5,5.4));x=range(3)
ax.bar([z-.19 for z in x],strict,width=.36,label='Strict originating-page Top-1')
ax.bar([z+.19 for z in x],exact,width=.36,label='Exact evidence Top-1')
ax.set_xticks(list(x),names);ax.set_ylim(0,100);ax.set_ylabel('Top-1 accuracy (%)')
ax.set_title('Phase 23: exact passage reprints change retrieval evaluation')
ax.legend(loc='upper right',fontsize=9)
for i in range(3):
 ax.text(i-.19,strict[i]+1.5,f'{strict[i]:.1f}%',ha='center',fontsize=9)
 ax.text(i+.19,exact[i]+1.5,f'{exact[i]:.1f}%',ha='center',fontsize=9)
fig.text(.12,.01,'35 repeated-passage keyword queries; 28 Imvo + 7 Isigidimi issues. Each newspaper weighted equally.',fontsize=8)
fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(O/'phase23_evaluation_gap.png',dpi=190)
print('WROTE',O/'phase23_evaluation_gap.png')
