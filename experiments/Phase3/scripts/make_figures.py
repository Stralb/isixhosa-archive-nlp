#!/usr/bin/env python3
"""Publication-style figures regenerated only from verified Phase3 summary CSV."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);arg=ap.parse_args()
out=Path(arg.output);d=pd.read_csv(out/'factorial_contrasts.csv')
rows=d[d.contrast=='allocation'].copy().sort_values(['period','family','view'])
rows['display']=rows.apply(lambda r:f"{r['family'].upper()} - {r['view'].upper()} view - {r['period']}",axis=1)
fig,ax=plt.subplots(figsize=(10.0,5.6))
y=np.arange(len(rows))
means=rows['mean'].to_numpy()
ax.errorbar(means,y,xerr=np.vstack((means-rows['ci_low'],rows['ci_high']-means)),fmt='o',capsize=4)
ax.set_yticks(y,rows['display']);ax.axvline(0,linestyle='--',linewidth=1)
ax.invert_yaxis();ax.set_xlabel('Word-like minus page-like view allocation (pieces / 100 letters)')
ax.set_title('Historical newspaper sampling: allocation effect by tokenizer and archive view')
ax.text(.0,-.14,'Paired 95% publication-issue bootstrap intervals; lower means more compact',transform=ax.transAxes,ha='left',va='center',fontsize=9)
fig.tight_layout(rect=(0,.06,1,1));fig.savefig(out/'phase3_allocation_effects.png',dpi=180);plt.close(fig)

sub=d[(d.contrast=='within_view') & (d.view=='en')].copy().sort_values(['period','family'])
sub['display']=sub.apply(lambda r:f"{r['family'].upper()} - {r['period']}",axis=1)
fig,ax=plt.subplots(figsize=(9.5,3.8))
y=np.arange(len(sub));means=sub['mean'].to_numpy()
ax.errorbar(means,y,xerr=np.vstack((means-sub['ci_low'],sub['ci_high']-means)),fmt='s',capsize=4)
ax.set_yticks(y,sub['display']);ax.invert_yaxis();ax.axvline(0,linestyle='--',linewidth=1)
ax.set_title('Secondary within-view selection effect on English-view text')
ax.set_xlabel('Word-conditional minus page-conditional (pieces / 100 letters)')
fig.tight_layout();fig.savefig(out/'phase3_within_view_effects.png',dpi=180);plt.close(fig)
print('CREATED FIGURES',flush=True)
