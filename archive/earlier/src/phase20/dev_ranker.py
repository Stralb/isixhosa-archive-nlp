"""Development-only ranker selection on Phase 19, NEVER a Phase 20 test metric."""
import json, numpy as np, regex as re, collections, math, time
from pathlib import Path
R=Path(__file__).resolve().parents[1]
D=[json.loads(x) for x in (R/'pages.jsonl').open()]
P=[json.loads(x) for x in (R/'phase19/phase19_per_query.jsonl').open()]
S=np.load(R/'phase19/phase19_bm25_score_matrices.npz')['raw']
ids={d['id']:i for i,d in enumerate(D)}
W=re.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
cache={}
def words(text):return [t.lower() for t in W.findall(text)]
def localmatch(q,d,w=120):
 """Maximum number of distinct query terms present in w consecutive page tokens."""
 qt=set(words(q)); qt={t for t in qt if len(t)>=3}
 if not qt:return 0
 if d not in cache:cache[d]=words(D[d]['raw'])
 seq=cache[d]
 poss=[(i,t) for i,t in enumerate(seq) if t in qt]
 counts=collections.defaultdict(int);left=0;best=0
 for right,(pos,t) in enumerate(poss):
  counts[t]+=1
  while poss[left][0]<pos-w+1:
   old=poss[left][1];counts[old]-=1
   if not counts[old]:del counts[old]
   left+=1
  best=max(best,len(counts))
 return best/len(qt)
def mean(b):
 return (float(np.mean([b[k] for k,z in enumerate(P) if z['source']==src])) for src in ('IMVO','ISIG'))
def score(ranks):
 vals={src:np.mean([int(P[k]['source_page']==D[int(ranks[k][0])]['id']) for k in range(len(P)) if P[k]['source']==src]) for src in ('IMVO','ISIG')}
 return (sum(vals.values())/2,vals)
res={}
for mode in ['max_local','bm25_local_025','bm25_local_05','bm25_local_10','bm25_local_20']:
 rank=[]
 for i,p in enumerate(P):
  cand=[ids[pid] for pid in p['candidate_pools']['raw_bm25']]
  if not cand:rank.append([]);continue
  local=np.array([localmatch(p['query'],j) for j in cand]);bm=S[i,cand]
  if mode=='max_local':key=local*100+bm/max(bm.max(),1e-6)*0.01
  else:
   coef=float(mode.rsplit('_',1)[-1])/100 if mode.endswith('_025') else {'bm25_local_05':.5,'bm25_local_10':1.,'bm25_local_20':2.}[mode]
   key=bm*(1+coef*local)
  rank.append([cand[j] for j in np.argsort(-key,kind='stable')])
 res[mode]=score(rank)
for name in ('raw_bm25','raw_fuzzy'):
 res[name]=(np.mean([np.mean([z[name]['anchor_top1'] for z in P if z['source']==src]) for src in ('IMVO','ISIG')]),{})
res['definition']='Phase19 DEV only. Primary Phase20 protocol will be frozen after selection and prior to Phase20 rank evaluation.'
(R/'phase20/dev_selection.json').write_text(json.dumps(res,indent=2))
print(json.dumps(res,indent=2))
