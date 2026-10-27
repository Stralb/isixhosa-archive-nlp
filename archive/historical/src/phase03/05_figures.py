#!/usr/bin/env python3
"""Phase 3 original data plots (no external images)."""
from pathlib import Path
import argparse
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(root):
 root=Path(root);out=root/'figures';out.mkdir(parents=True,exist_ok=True)
 dev=pd.read_csv(root/'outputs/dev/risk_coverage.csv')
 test=pd.read_csv(root/'outputs/test/risk_coverage.csv')
 methods=['POOLED','SOURCE_HALF_STD','EST_IPW','RIDGE_IMPUTE','AIPW']
 dn=dev[(dev.scenario=='MNAR_LATENT')&(dev.world=='INJECTED')&(dev.requested_coverage==.2)]
 tn=test[(test.scenario=='MNAR_LATENT')&(test.world=='INJECTED')&(test.requested_coverage==.2)]
 dv=[100*float(dn[dn.method==m].wrong_call_rate.iloc[0]) for m in methods]
 tv=[100*float(tn[tn.method==m].wrong_call_rate.iloc[0]) for m in methods]
 xx=np.arange(len(methods))
 fig,ax=plt.subplots(figsize=(9.7,5.0))
 ax.bar(xx-.2,dv,.38,label='1884 to 1885 (dev)')
 ax.bar(xx+.2,tv,.38,label='1887 to 1888 (test)')
 ax.set_xticks(xx,methods,rotation=22,ha='right')
 ax.set_ylabel('Incorrect sign decisions (%)')
 ax.set_title('Unknown-text archive loss: development to later-year test\nAt matched 20% decision coverage, injected world')
 ax.legend();ax.grid(axis='y',alpha=.2)
 fig.tight_layout();fig.savefig(out/'phase3_mnar_dev_test_risk.png',dpi=160);plt.close(fig)
 d=pd.read_csv(root/'outputs/test/summary_metrics.csv')
 scenarios=['MAR_SOURCE_PAGE','MAR_LENGTH_DATE','MNAR_LATENT','ISSUE_DROPOUT','ZERO_CELL','REPEAT_PREF']
 q=d[(d.world=='INJECTED')&(d['class']=='point')]
 xx=np.arange(len(scenarios));labels=['Source/page MAR','Length/date MAR','Hidden-text MNAR','Whole issue loss','Absent source cell','Repeated-text loss']
 fig,ax=plt.subplots(figsize=(10.0,5.0))
 for off,m,lab in [(-.17,'POOLED','Pooled'),(.17,'AIPW','Doubly robust (MAR assumption)')]:
  vals=[100*float(q[(q.scenario==s)&(q.method==m)].wrong_call_rate_among_decisions.iloc[0]) for s in scenarios]
  ax.bar(xx+off,vals,.34,label=lab)
 ax.set_xticks(xx,labels,rotation=20,ha='right');ax.set_ylabel('Incorrect decisions among issued calls (%)')
 ax.set_title('Selection mechanism governs reliability on 1887-1888 text')
 ax.legend();ax.grid(axis='y',alpha=.2)
 fig.tight_layout();fig.savefig(out/'phase3_heldout_wrong_calls.png',dpi=160);plt.close(fig)
 b=d[(d.scenario=='MNAR_LATENT')&(d.world=='INJECTED')&(d['class']=='identification_set')]
 labs=['Unrestricted\nManski','Gap <=5/10k','Gap <=20/10k']
 b=b.set_index('method').loc[['MANSKI','GAMMA_0005','GAMMA_0020']]
 fig,ax=plt.subplots(figsize=(8.5,4.8))
 xx=np.arange(3);ax.bar(xx-.18,b.decision_coverage.values*100,.35,label='Sign decisions (%)')
 ax.bar(xx+.18,b.effect_in_interval.values*100,.35,label='Complete-archive target within bound (%)')
 ax.set_xticks(xx,labs);ax.set_ylim(0,110)
 ax.set_ylabel('Percent of simulation-feature evaluations')
 ax.set_title('Missing-text assumptions: more answers, weaker guarantees\n1887-1888 hidden-text selection (INJECTED)')
 ax.grid(axis='y',alpha=.2);ax.legend()
 fig.tight_layout();fig.savefig(out/'phase3_sensitivity_identification.png',dpi=160);plt.close(fig)
 # Portable summary from both splits, so all values can be re-plotted without model runs.
 rows=[]
 for split in ('dev','test'):
  df=pd.read_csv(root/'outputs'/split/'summary_metrics.csv')
  for scenario in scenarios:
   for world in ('NATURAL','INJECTED'):
    for method in ('POOLED','SOURCE_HALF_STD','EST_IPW','AIPW','KNOWN_HT_ORACLE','MANSKI','GAMMA_0005','GAMMA_0020'):
     row=df[(df.scenario==scenario)&(df.world==world)&(df.method==method)].iloc[0]
     rows.append({'split':split,'scenario':scenario,'world':world,'method':method,
                  'wrong_call_rate':row.wrong_call_rate_among_decisions,
                  'coverage':row.decision_coverage,'mae_per_10000':row.mean_absolute_error_per_10000,
                  'contains_true_effect':row.effect_in_interval,'assumption_holds':row.premise_holds})
 pd.DataFrame(rows).to_csv(root/'outputs/dev_test_comparison.csv',index=False)
 print('FIGURES written',out)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();main(a.root)
