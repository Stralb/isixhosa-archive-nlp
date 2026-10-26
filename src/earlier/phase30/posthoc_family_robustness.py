#!/usr/bin/env python3
"""Post-result sensitivity: approximate orthographic-family clustering, not morphology labels."""
from pathlib import Path
import csv,json,collections
import numpy as np
from scipy.stats import spearmanr
D=Path(__file__).resolve().parent
R=list(csv.DictReader((D/'shared_word_frequency_vs_pieces.csv').open()))
rows=[]
for seed in range(3):
 a=[r for r in R if int(r['seed'])==seed]
 gr=collections.defaultdict(list)
 for r in a:gr[r['word'][:4]].append(r)
 keys=sorted(gr);rng=np.random.default_rng(30724+seed);samples=[]
 for _ in range(2000):
  arr=[z for k in rng.choice(keys,len(keys),replace=True) for z in gr[k]]
  x=np.array([float(z['train_frequency_log_ratio']) for z in arr]);y=np.array([int(z['isig_minus_imvo_pieces']) for z in arr]);
  if np.std(y)>0:samples.append(float(spearmanr(x,y).statistic))
 ci=np.quantile(samples,[.025,.975])
 rows.append(dict(seed=seed,n_orthographic_prefix_groups=len(keys),mean_rho_bootstrap=float(np.mean(samples)),orthographic_family_bootstrap_95ci=list(map(float,ci)),bootstrap_low_positive=bool(ci[0]>0)))
(D/'POSTHOC_ORTHOGRAPHIC_FAMILY_SENSITIVITY.json').write_text(json.dumps({'classification':'post-hoc robustness, not a frozen primary test','clustering':'first four characters is an operational string proxy, not validated morphology','seeds':rows},indent=2))
print(json.dumps(rows,indent=2))
