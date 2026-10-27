"""Paired same-source-page comparison of original flattened-word and new paragraph-valid query sets."""
from pathlib import Path
import json,numpy as np,collections
P=Path('/mnt/data/isiXhosa_three_track_pilot');O=Path('/mnt/data/isiXhosa_phase15_source_grounded_search')
old={(r['qid'],r['kind'],r['method']):r for r in (json.loads(x) for x in open(P/'retrieval_repaired_results.jsonl'))}
new={(r['qid'],r['kind'],r['method']):r for r in (json.loads(x) for x in open(O/'paragraph_retrieval_results.jsonl'))}
Q=[json.loads(x) for x in open(O/'paragraph_queries.jsonl')];out={}
for kind in ('clean','noisy'):
 for metric in ('exact_r1','any_r1'):
  v='exact1' if metric=='exact_r1' else 'any1'; d=[]
  for q in Q:
   o=old[q['id'],kind,'raw']['exact_r1' if v=='exact1' else 'any_r1'];n=new[q['id'],kind,'raw'][v];d.append({'qid':q['id'],'issue':q['issue'],'src':q['source'],'old':o,'new':n,'delta':n-o})
  D=np.array([[r['old'],r['new'],r['delta']] for r in d],dtype=float)
  # Bootstrap matched issue clusters with within-source stratification
  G=collections.defaultdict(list)
  for r in d:G[(r['src'],r['issue'])].append(r)
  sims=np.zeros(10000);den=np.zeros(10000);rng=np.random.default_rng(20261023)
  for src in ('IMVO','ISIG'):
   gs=[v for (s,i),v in G.items() if s==src];sm=np.array([sum(r['delta'] for r in g) for g in gs]);sz=np.array([len(g) for g in gs])
   j=rng.integers(len(gs),size=(10000,len(gs)));sims+=sm[j].sum(axis=1);den+=sz[j].sum(axis=1)
  val=sims/den
  out[kind+'_'+metric]={'n':len(d),'old_mean':round(D[:,0].mean(),6),'new_mean':round(D[:,1].mean(),6),'paired_delta_pp':round(100*D[:,2].mean(),3),
     'issue_bootstrap_CI95_delta_pp':[round(float(x),3) for x in 100*np.quantile(val,[.025,.975])]}
(O/'old_new_paired_retrieval.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
