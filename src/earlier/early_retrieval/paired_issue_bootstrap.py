"""Paired cluster uncertainty and source-aware diagnostic breakdown."""
from pathlib import Path
import json, numpy as np, collections, statistics
O=Path('/mnt/data/isiXhosa_three_track_pilot')
R=[json.loads(s) for s in (O/'retrieval_repaired_results.jsonl').open()]
Q={q['id']:q for q in (json.loads(s) for s in (O/'query_repaired.jsonl').open())}
D={(r['method'],r['kind'],r['qid']):r for r in R}
seq=list(Q);groups=collections.defaultdict(list)
for q in Q.values():groups[(q['source'],q['lang'])].append(q)

def delta(method_a,method_b,metric,kind,eligible,seed=20261023,rep=4000):
 # Positive = method_a better than method_b (recall@1 etc.)
 dataset=[Q[k] for k in seq if eligible(Q[k])]
 byissue=collections.defaultdict(list)
 for q in dataset:
  ia=D[(method_a,kind,q['id'])][metric]; ib=D[(method_b,kind,q['id'])][metric]
  byissue[q['issue']].append(ia-ib)
 blocks=[np.array(y,dtype=float) for y in byissue.values()]
 rng=np.random.default_rng(seed)
 sums=np.array([x.sum() for x in blocks],dtype=np.float64)
 lengths=np.array([len(x) for x in blocks],dtype=np.float64)
 pick=rng.integers(0,len(blocks),size=(rep,len(blocks)))
 samples=sums[pick].sum(axis=1)/lengths[pick].sum(axis=1)
 del pick
 return {'n_queries':len(dataset),'issues':len(blocks),'delta_pp':round(float(100*np.mean([x for v in byissue.values() for x in v])),3),
         'ci95_pp':[round(float(100*x),3) for x in np.quantile(samples,[.025,.975])]}

checks=[
 ('fielded_auto_LID','raw','exact_r1','noisy','all',lambda q:True),
 ('fielded_auto_LID','raw','any_r1','noisy','all',lambda q:True),
 ('fielded_oracle','raw','exact_r1','noisy','all',lambda q:True),
 ('fielded_auto_LID','raw','exact_r1','clean','all',lambda q:True),
 ('fielded_auto_LID','raw','exact_r1','noisy','xh_only',lambda q:q['lang']=='xh'),
 ('fielded_auto_LID','raw','exact_r1','noisy','en_only',lambda q:q['lang']=='en'),
 ('Xh_only','raw','exact_r1','noisy','xh_only',lambda q:q['lang']=='xh'),
 ('En_only','raw','exact_r1','noisy','en_only',lambda q:q['lang']=='en'),
 ('raw_filter_en_multitag','raw_filter_en_majority','exact_r1','clean','en_only',lambda q:q['lang']=='en'),
 ('raw_filter_en_multitag','raw_filter_en_majority','any_r1','clean','en_only',lambda q:q['lang']=='en'),
 ('raw','raw','any_r1','clean','original_n',lambda q:True),
]
res={}
for a,b,metric,kind,label,selector in checks:
 if a==b:continue
 k=f'{a} - {b} | {metric} | {kind} | {label}'
 res[k]=delta(a,b,metric,kind,selector)
 print(k,res[k],flush=True)
for src in ('IMVO','ISIG'):
 for lang in ('xh','en'):
  sel=lambda q:q['source']==src and q['lang']==lang
  for method in ('fielded_auto_LID','fielded_oracle'):
   k=f'{method} - raw | noisy exact_r1 | {src}_{lang}'
   res[k]=delta(method,'raw','exact_r1','noisy',sel)
   print(k,res[k],flush=True)
# Raw exact-page vs any-evidence-page difference (deterministic per item but CI issues descriptive)
for kind in ('clean','noisy'):
 samples=[]
 for q in Q.values():
  d=D[('raw',kind,q['id'])]
  samples.append(dict(qid=q['id'],source=q['source'],lang=q['lang'],issue=q['issue'],ambiguous=bool(q['num_exact_match_pages']>1),
                      exact=d['exact_r1'],any=d['any_r1'],num=q['num_exact_match_pages']))
 summary={}
 for tag,zz in [('all',samples),('unique',[z for z in samples if not z['ambiguous']]),('multi',[z for z in samples if z['ambiguous']])]:
  summary[tag]={'n':len(zz),'exact_r1':float(np.mean([z['exact'] for z in zz])),'any_r1':float(np.mean([z['any'] for z in zz])),
                'multi_ref_frac':float(np.mean([z['ambiguous'] for z in zz]))}
 res[f'raw_{kind}_ambiguity_breakdown']=summary
 print(kind,summary,flush=True)
(O/'paired_issue_bootstrap_results.json').write_text(json.dumps(res,indent=2))
