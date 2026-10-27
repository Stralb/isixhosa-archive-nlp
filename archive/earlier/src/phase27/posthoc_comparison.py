#!/usr/bin/env python3
"""Phase27 posthoc model-family contrasts from frozen predictions. Not new confirmation."""
from pathlib import Path
import csv
import numpy as np
import pandas as pd
D=Path(__file__).parent
P=pd.read_csv(D/'period_predictions.csv')
issue=P.groupby(['masked','model','seed','train_source','test_source','year','issue','month'],as_index=False).agg(acc=('correct','mean'))
S={}
for k,g in issue.groupby(['masked','model','seed','train_source','test_source']):
    num=np.zeros((2,12),dtype=float)
    den=np.zeros((2,12),dtype=float)
    for r in g.itertuples():
        yi=0 if r.year==1885 else 1; mi=r.month-1
        num[yi,mi]+=r.acc;den[yi,mi]+=1
    S[k]=(num,den)

def cell(mask,model,train,test,weights):
    seeds=[-1] if model in ('char','word') else [27,28,29]
    vals=[]
    for seed in seeds:
        num,den=S[mask,model,seed,train,test]
        vals.append(.5*sum(np.dot(weights,num[y])/np.dot(weights,den[y]) for y in (0,1)))
    return float(np.mean(vals))

def transfer(mask,model,weights):
    return .5*(cell(mask,model,'IMVO','ISIG',weights)+cell(mask,model,'ISIG','IMVO',weights))

rng=np.random.default_rng(27112)
weights=[np.ones(12,dtype=int)]+[np.bincount(rng.integers(0,12,size=12),minlength=12) for _ in range(2000)]
rows=[]
for masked in (0,1):
    for other in ('word','bpe0','bpe50','bpe100'):
        draws=[transfer(masked,'char',w)-transfer(masked,other,w) for w in weights]
        rows.append(dict(masked=masked,other=other,
                         char_minus_other_cross_accuracy=draws[0],
                         ci_low=float(np.quantile(draws[1:],.025)),
                         ci_high=float(np.quantile(draws[1:],.975)),
                         label='POSTHOC'))
with open(D/'posthoc_model_family.csv','w',newline='',encoding='utf8') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
print(pd.DataFrame(rows).to_string(index=False))
