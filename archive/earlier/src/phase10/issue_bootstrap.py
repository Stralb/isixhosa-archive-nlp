#!/usr/bin/env python3
"""Conditioned, matched neural-seed and newspaper-issue cluster resampling for Phase 10.

This is descriptive uncertainty only: 1887 validation has 12 Imvo issues and 3 Isigidimi issues.
"""
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
SEEDS=(101,202,303)
PAIRS={'5001_ng_minus_startn':('imvo5001_greedy','imvo5001_second'),
 '5004_ng_minus_startn':('imvo5004_second','imvo5004_greedy'),
 '5012_ng_minus_startn':('imvo5012_greedy','imvo5012_second'),
 '5004_selected_minus_fullpool':('imvo5004_second','fullpool_imvo')}
rng=np.random.default_rng(20261023)
result={}
for label,(a,b) in PAIRS.items():
    rows=[]
    for model in (a,b):
        modes=[]
        for seed in SEEDS:
            j=json.loads((ROOT/f'run_{model}_{seed}.json').read_text())
            one={}
            for src in ('IMVO','ISIG'):
                ix=defaultdict(lambda:[0.0,0])
                for doc in j['evaluation'][src]['doc']:
                    issue=doc['id'].rsplit('_p',1)[0]
                    ix[issue][0]+=doc['nats'];ix[issue][1]+=doc['chars']
                one[src]=ix
            modes.append(one)
        rows.append(modes)
    sources={}
    for src in ('IMVO','ISIG'):
        keys=sorted(rows[0][0][src]);assert len(keys)==(12 if src=='IMVO' else 3),len(keys)
        for m in range(2):
            for s in range(3): assert keys==sorted(rows[m][s][src])
        mat=np.empty((2,3,len(keys),2),dtype=np.float64)
        for m in range(2):
            for s in range(3):
                for ki,k in enumerate(keys):mat[m,s,ki,:]=rows[m][s][src][k]
        sources[src]=mat
    point=[]
    for s in range(3):
        vals={}
        for src,z in sources.items():
            score=z[:,s,:,0].sum(axis=1)/z[:,s,:,1].sum(axis=1)/math.log(2)
            vals[src]=score
        point.append({'IMVO':(vals['IMVO'][0]-vals['IMVO'][1]),'ISIG':(vals['ISIG'][0]-vals['ISIG'][1]),
                'macro':(vals['IMVO'].mean()-vals['ISIG'].mean())*0 # overwritten below
                })
        point[-1]['macro']=(vals['IMVO'][0]+vals['ISIG'][0]-vals['IMVO'][1]-vals['ISIG'][1])/2
        point[-1]['worst']=max(vals['IMVO'][0],vals['ISIG'][0])-max(vals['IMVO'][1],vals['ISIG'][1])
    boot={key:[] for key in ('IMVO','ISIG','macro','worst')}
    for _ in range(10000):
        indseed=rng.choice(3,size=3,replace=True)
        stats={}
        for src,z in sources.items():
            indissue=rng.choice(z.shape[2],size=z.shape[2],replace=True)
            x=z[:,indseed,:,:][:,:,indissue,:].sum(axis=2)
            # 2 policies x 3 sampled neural seeds, page denominator by sampled issue groups
            stats[src]=(x[:,:,0]/x[:,:,1]/math.log(2)).mean(axis=1)
            boot[src].append(float(stats[src][0]-stats[src][1]))
        boot['macro'].append(float(((stats['IMVO']+stats['ISIG'])/2)[0]-((stats['IMVO']+stats['ISIG'])/2)[1]))
        boot['worst'].append(float(max(stats['IMVO'][0],stats['ISIG'][0])-max(stats['IMVO'][1],stats['ISIG'][1])))
    result[label]={'a':a,'b':b,'per_neural_seed':point,
             'paired_mean':{k:float(np.mean([r[k] for r in point])) for k in point[0]},
             'paired_issue_and_seed_bootstrap_95':{k:np.quantile(boot[k],[.025,.975]).tolist() for k in boot},
             'counts':{'IMVO_issues':12,'ISIG_issues':3,'neural_seeds':3},
             'note':'conditional descriptive CI, few ISIG issues and all corpora previously inspected'}
(ROOT/'ISSUE_BOOTSTRAP.json').write_text(json.dumps(result,indent=2))
for k,v in result.items():print(k,'worst',v['paired_mean']['worst'],'95',v['paired_issue_and_seed_bootstrap_95']['worst'])
