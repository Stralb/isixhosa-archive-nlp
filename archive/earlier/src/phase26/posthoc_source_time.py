#!/usr/bin/env python3
"""POST-HOC: source-versus-year differences under equal 12-issue/month sampling."""
import collections,csv,json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
D=Path(__file__).parent
W=[json.loads(l) for l in open(D/'windows.jsonl')]
issues=collections.defaultdict(lambda:dict(source=None,year=None,month=None,tokens=[]))
for w in W:
    if w['year'] not in (1885,1888):continue
    rec=issues[w['issue']];rec['source']=w['source'];rec['year']=w['year'];rec['month']=w['month'];rec['tokens']+=w['tokens']
key=list(sorted(issues))
counters=[]
for k in key:
    c=collections.Counter()
    for w in issues[k]['tokens']:
        for i in range(len(w)-3):c[w[i:i+4]]+=1
    counters.append(c)
X=normalize(DictVectorizer().fit_transform(counters),norm='l1',axis=1).tocsr()
Ix={k:i for i,k in enumerate(key)}
by=collections.defaultdict(list)
for k,rec in issues.items():by[(rec['source'],rec['year'],rec['month'])].append(k)
print('month stratum counts',[(s,y,len([k for k in key if issues[k]['source']==s and issues[k]['year']==y]),len(set(issues[k]['month'] for k in key if issues[k]['source']==s and issues[k]['year']==y))) for s in ('IMVO','ISIG') for y in (1885,1888)])
def ave(chosen):return np.asarray(X[[Ix[k] for k in chosen]].mean(axis=0)).ravel()
def js(a,b):return float(jensenshannon(a,b,base=2)**2)
rng=np.random.default_rng(20261024)
rows=[]
for seed in range(100):
    av={}
    for s in ('IMVO','ISIG'):
        for y in (1885,1888):
            selected=[]
            for m in range(1,13):
                opts=by[(s,y,m)]
                if not opts:
                    raise RuntimeError((s,y,m))
                selected.append(opts[int(rng.integers(0,len(opts)))])
            assert len(selected)==12
            av[(s,y)]=ave(selected)
    rows.append(dict(replicate=seed,year_change_IMVO=js(av['IMVO',1885],av['IMVO',1888]),year_change_ISIG=js(av['ISIG',1885],av['ISIG',1888]),paper_gap_1885=js(av['IMVO',1885],av['ISIG',1885]),paper_gap_1888=js(av['IMVO',1888],av['ISIG',1888])))
with open(D/'posthoc_source_time_js.csv','w',newline='') as f:
    c=csv.DictWriter(f,fieldnames=rows[0].keys());c.writeheader();c.writerows(rows)
print('mean', {k:np.mean([r[k] for r in rows]) for k in rows[0] if k!='replicate'})
print('median', {k:np.median([r[k] for r in rows]) for k in rows[0] if k!='replicate'})
