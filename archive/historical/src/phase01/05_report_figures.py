#!/usr/bin/env python3
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=Path(__file__).resolve().parents[1]/'outputs'
with (p/'method_comparison.csv').open() as f:rows=list(csv.DictReader(f))
with (p/'oracle_comparison.csv').open() as f:oracle=list(csv.DictReader(f))
scenarios=['SOURCE','PAGE'];methods=['POOLED','SOURCE','SOURCE_PAGE','ORACLE']
for metric,output,title,ylabel in [
 ('mean_BH_discoveries','figure_apparent_discoveries.png','Apparent change discoveries under known-null pseudo-periods','Mean BH discoveries out of 200 forms'),
 ('median_over_reps_of_feature_median_abs_per10k','figure_effect_sizes.png','Magnitude of apparent change across sampling policies','Median |B - A| per 10,000 words')]:
 fig,ax=plt.subplots(figsize=(9.5,4.7))
 series=[]
 for meth in methods:
  if metric.startswith('median') and meth=='ORACLE':continue
  values=[]
  for scenario in scenarios:
   if meth=='ORACLE':r=next(x for x in oracle if x['scenario']==scenario)
   else:r=next(x for x in rows if x['scenario']==scenario and x['method']==meth)
   values.append(float(r[metric]))
  series.append((meth,values))
 w=.18;x=np.arange(2)
 for i,(meth,values) in enumerate(series):
  positions=x+(i-(len(series)-1)/2)*w
  bb=ax.bar(positions,values,width=w,label=meth.replace('_',' + '))
  for rect,v in zip(bb,values):ax.text(rect.get_x()+rect.get_width()/2,rect.get_height()+.04,f'{v:.2f}',ha='center',fontsize=9)
 ax.set_xticks(x,labels=['Newspaper mix shifted','Page-position mix shifted']);ax.set_ylabel(ylabel)
 ax.set_title(title);ax.legend(frameon=False,ncols=4,loc='upper right');ax.spines[['top','right']].set_visible(False)
 ax.set_ylim(bottom=0,top=max(v for _,values in series for v in values)*1.23)
 fig.text(.08,.012,'200 simulations per scenario; 200 common wordforms. No real time or semantic change is asserted.',fontsize=8,alpha=.8)
 fig.tight_layout(rect=(0,.055,1,1));fig.savefig(p/output,dpi=170);plt.close(fig)
print('Created figures')
