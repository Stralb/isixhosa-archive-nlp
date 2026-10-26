"""Resampling edition and tokenizer-seed variation separately, using frozen scores."""
import json,re
from pathlib import Path
from collections import defaultdict
import numpy as np

base=Path(__file__).resolve().parent
score=json.loads((base/'modern_external_scores_frozen.json').read_text())['document_results']['vukuzenzele_xho']
seeds=(11,23,37,47,59)
rng=np.random.default_rng(20261022)
byed=defaultdict(list)
for r in score:
    e=r['id'].rsplit('_',1)[-1]
    cr={x['imvo_share']:x for x in r['costs']}
    for metric,key in (('unseen','supported_unseen_E_by_seed'),('all','supported_E_by_seed')):
        diff=np.array(cr[1][key])-np.array(cr[.5][key])
        byed[e].append((metric,diff,r['alpha_chars'],r['unseen_alpha_chars']))

result={}
for metric in ['unseen','all']:
    g=[];counts=[]
    for e,rs in sorted(byed.items()):
        subset=[d for m,d,_,_ in rs if m==metric]
        g.append(np.sum(subset,axis=0));counts.append(len(subset))
    mat=np.asarray(g)
    denom=np.asarray(counts)
    per_seed=mat.sum(axis=0)/denom.sum()
    B=30000;n_ed=len(mat);n_seed=len(seeds)
    edition_samples=rng.integers(0,n_ed,size=(B,n_ed))
    seed_samples=rng.integers(0,n_seed,size=(B,n_seed))
    # Two-way bootstrap of edition clusters and seed draws.
    result_boot=np.empty(B)
    for i in range(B):
        eix=edition_samples[i]; six=seed_samples[i]
        result_boot[i]=mat[eix][:,six].sum()/(denom[eix].sum()*n_seed)
    # Seed-only bootstrap for extra transparency.
    s=rng.integers(0,n_seed,size=(B,n_seed))
    bs_seed=per_seed[s].mean(axis=1)
    result[metric]={'n_editions':n_ed,'n_model_seeds':n_seed,'mean_difference':float(per_seed.mean()),
       'per_seed_diff':[float(x) for x in per_seed],
       'edition_and_seed_bootstrap_ci_95':[float(x) for x in np.quantile(result_boot,[.025,.975])],
       'seed_only_bootstrap_ci_95':[float(x) for x in np.quantile(bs_seed,[.025,.975])],
       'fraction_bootstrap_draws_with_benefit':float(np.mean(result_boot<0)),
       'warning':'Bootstrap intervals describe variability over observed editions and five seeds; neither justifies population generalisation.'}
output=base/'modern_seed_uncertainty.json'
output.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
