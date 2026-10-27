from pathlib import Path
from collections import defaultdict
import json,numpy as np
O=Path('/mnt/data/isiXhosa_phase18_evidence_routing')
q=[*map(json.loads,open(O/'balanced_en_queries.jsonl')),*map(json.loads,open(O/'balanced_xh_queries.jsonl'))]
r=[*map(json.loads,open(O/'balanced_query_results.jsonl'))]
src=np.array([x['source'] for x in r]);view=np.array([x['view'] for x in r]);test=np.array([x['split']=='test' for x in r]);unique=np.array([x['unique'] for x in r]);front=np.array([x['front'] for x in r]);
H=[json.loads(s) for s in open(O/'hard_fuzzy_results.jsonl')]
K=['clean','char_only','original_char_boundary','char_plus_boundary','two_char_plus_boundary']
assert len(H)==len(q)*len(K)
by={key:[] for key in K}
for z in H:by[z['cohort']].append(z)
for k in K:assert len(by[k])==len(q) and [x['qid'] for x in by[k]]==[z['id'] for z in q]

def macro(a,valid):return float(np.mean([a[valid&(src==s)&(view==lg)].mean() for s in ('IMVO','ISIG') for lg in ('en','xh')]))

rng=np.random.default_rng(20261023);B=10000;group={s:defaultdict(list) for s in ('IMVO','ISIG')}
for i in np.where(test)[0]:group[src[i]][q[i]['issue']].append(i)
chunks={s:[np.array(a,dtype=int) for a in group[s].values()] for s in group}
indices={s:rng.integers(0,len(chunks[s]),size=(B,len(chunks[s]))) for s in chunks}

def issue_boot(delta):
 cells=[]
 for s in ('IMVO','ISIG'):
  for lg in ('en','xh'):
   n=np.array([delta[ch][view[ch]==lg].sum() for ch in chunks[s]])
   d=np.array([(view[ch]==lg).sum() for ch in chunks[s]])
   id=indices[s]
   cells.append(n[id].sum(axis=1)/np.maximum(d[id].sum(axis=1),1))
 dist=np.mean(cells,axis=0)
 return {'point':macro(delta,test),'ci95':list(map(float,np.quantile(dist,[.025,.975])))}

result={}
for name in K:
 vals={k:np.array([z[k] for z in by[name]],dtype=float) for k in ('raw_fuzzy','hard_fuzzy','gated_fuzzy','oracle_view_fuzzy')}
 coh={'top1_macro':{k:macro(v,test) for k,v in vals.items()},'top1_unique_macro':{k:macro(v,test&unique) for k,v in vals.items()},'top1_nonfront_macro':{k:macro(v,test&(~front)) for k,v in vals.items()},'comparisons':{}}
 for a,b in [('gated_fuzzy','raw_fuzzy'),('gated_fuzzy','hard_fuzzy'),('hard_fuzzy','raw_fuzzy'),('gated_fuzzy','oracle_view_fuzzy')]:coh['comparisons'][a+'_minus_'+b]=issue_boot(vals[a]-vals[b])
 result[name]=coh
print(json.dumps(result,indent=2))
(O/'FINAL_COMPARATIVE_UNCERTAINTY.json').write_text(json.dumps(result,indent=2))
