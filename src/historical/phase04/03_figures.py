#!/usr/bin/env python3
"""Reproduce preregistered Phase4 diagnostic figures from Phase4 test CSV summaries."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LAB={'RANDOM':'Random missing pages','WIDTH_GREEDY':'Minimize worst-case width','CELL_REPAIR':'Restore empty cells first'}

def draw(folder,out):
 folder=Path(folder);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 data=pd.read_csv(folder/'scenario_summary.csv')
 b=data[data.world=='INJECTED']
 rates=b.groupby(['policy','budget'],as_index=False)[['mean_width_10k','mean_sign_coverage']].mean()
 for key,ylabel,filename in [('mean_width_10k','Identification-set width (per 10,000 words)','phase4_audit_budget_width.png'),
                            ('mean_sign_coverage','Fraction of defensible sign conclusions','phase4_audit_budget_sign_calls.png')]:
  fig,ax=plt.subplots(figsize=(8.4,5.1))
  for policy in LAB:
   row=rates[rates.policy==policy].sort_values('budget')
   ax.plot(row.budget*100,row[key],marker='o',linewidth=2,label=LAB[policy])
  ax.set_xlabel('Fraction of originally missing pages hypothetically recovered (%)')
  ax.set_ylabel(ylabel)
  ax.set_title('Held-out 1887-1888: assumed exact page recovery (injected world)')
  ax.grid(alpha=.25)
  ax.legend(fontsize=8,frameon=False)
  ax.set_xlim(0,100)
  if key=='mean_sign_coverage':ax.set_ylim(bottom=0)
  fig.tight_layout();fig.savefig(out/filename,dpi=160);plt.close(fig)
 zero=b[(b.scenario=='ZERO_CELL')]
 fig,ax=plt.subplots(figsize=(8.4,5.1))
 for policy in LAB:
  row=zero[zero.policy==policy].sort_values('budget')
  ax.plot(row.budget*100,row.mean_empty_cells,marker='o',linewidth=2,label=LAB[policy])
 ax.set_xlabel('Fraction of missing pages hypothetically recovered (%)')
 ax.set_ylabel('Mean wholly unrepresented source-page cells remaining')
 ax.set_title('Restoring a cell: positivity support does not guarantee identification')
 ax.set_xlim(0,100);ax.set_ylim(bottom=0);ax.grid(alpha=.25);ax.legend(fontsize=8,frameon=False)
 fig.tight_layout();fig.savefig(out/'phase4_zero_cell_support.png',dpi=160);plt.close(fig)
 print('figures complete',len(list(out.glob('phase4_*.png'))))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--test',required=True);a.add_argument('--out',required=True);z=a.parse_args();draw(z.test,z.out)
