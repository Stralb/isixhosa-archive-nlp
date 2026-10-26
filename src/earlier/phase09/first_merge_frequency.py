#!/usr/bin/env python3
"""Diagnostic initial-pair frequencies on frozen sample multisets; no validation reads."""
import collections,json
from pathlib import Path
import run_phase9 as p
ROOT=p.ROOT
rows=[]
for mix in (.5,1.):
 for seed in p.SEEDS:
    sample=p.get_sample(mix,seed)
    counts=collections.Counter()
    for w in sample:
        tok=['▁',*w]
        counts.update(zip(tok,tok[1:]))
    rank=sorted(counts.items(), key=lambda item:(-item[1],item[0]))
    top=rank[:8]
    pivot_ng=counts[('n','g')];pivot_n=counts[('▁','n')]
    model=json.loads((p.MODELS/f'classic_mix{int(mix*100):03d}_draw{seed}.json').read_text())
    row={'mix':mix,'seed':seed,'first_merge':model['merges'][0],
         'pair_ng_count':pivot_ng,'pair_wordstart_n_count':pivot_n,
         'ng_minus_wordstart_n':pivot_ng-pivot_n,
         'top_margin':top[0][1]-top[1][1],
         'top_pairs':[{'pair':list(pair),'count':count} for pair,count in top]}
    assert tuple(row['first_merge'])==top[0][0]
    rows.append(row)
    print('PAIRCOUNTS',mix,seed,'top',tuple(row['first_merge']),
      'ng',pivot_ng,'wordstart_n',pivot_n,'gap',pivot_ng-pivot_n,'topmargin',row['top_margin'])
(ROOT/'first_merge_frequency_results.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
print('AGGREGATE')
for mix in (.5,1.):
 r=[x for x in rows if x['mix']==mix]
 print(mix,'first merge counts',dict(collections.Counter(''.join(x['first_merge']) for x in r)),
       'pivot diff range',min(x['ng_minus_wordstart_n'] for x in r),max(x['ng_minus_wordstart_n'] for x in r),
       'mean gap',sum(x['ng_minus_wordstart_n'] for x in r)/len(r))
