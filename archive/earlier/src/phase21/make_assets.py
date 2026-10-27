"""Create read-only Phase21 result tables and visualizations from completed frozen results."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
z=json.loads((R/'phase21_results.json').read_text())
methods=z['methods'];forms=z['forms']
with (R/'phase21_summary.csv').open('w',newline='') as f:
 writer=csv.writer(f);writer.writerow(['form','method','metric','macro_percent','ci95_low_percent','ci95_high_percent','imvo_percent','isigidimi_percent'])
 for form in forms:
  for method in methods:
   for met in ('recall20','top1','top5','origin_top1'):
    v=z['primary_metrics'][form][method][met]
    writer.writerow([form,method,met,*[round(100*x,3) for x in (v['macro'],*v['ci95'],v['by_source']['IMVO'],v['by_source']['ISIG'])]])
with (R/'phase21_hypotheses.csv').open('w',newline='') as f:
 writer=csv.writer(f);writer.writerow(['hypothesis','contrast_percentage_points','ci95_low','ci95_high','imvo_pp','isigidimi_pp','passed'])
 for name,v in z['hypotheses'].items():
  writer.writerow([name,*[round(100*x,3) for x in (v['macro'],*v['ci95'],v['by_source']['IMVO'],v['by_source']['ISIG'])],v['pass']])
for metric,subset,fn,title in [
 ('recall20',['word20','char20','hybrid15w5c','xh20_fuzzy'],'phase21_candidate_recall.png','Known-evidence candidate recall at Top-20'),
 ('top1',['word20','word20_fuzzy','char20_fuzzy','hybrid15w5c_fuzzy','xh20_fuzzy'],'phase21_top1.png','Known-evidence Top-1 retrieval')]:
 x=np.arange(len(forms));w=.16
 fig,ax=plt.subplots(figsize=(11,5.3))
 for i,met in enumerate(subset):
  vals=[100*z['primary_metrics'][form][met][metric]['macro'] for form in forms]
  ax.bar(x+(i-(len(subset)-1)/2)*w,vals,width=w,label=met)
 ax.set_xticks(x,['Keyword','Clean','1 edit','2 edits + boundary','3 edits + boundary'])
 ax.set_ylabel('Equal-newspaper macro (%)');ax.set_ylim(0,105)
 ax.set_title(title+' | 60 paired historical issues')
 ax.legend(loc='upper right',frameon=False,fontsize=9)
 ax.grid(axis='y',alpha=.2)
 fig.tight_layout();fig.savefig(R/fn,dpi=170);plt.close(fig)
print('Phase21 result tables and separate figures regenerated')
