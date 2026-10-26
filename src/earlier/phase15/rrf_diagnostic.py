"""RRF exploratory comparison; unaltered fixed k=60 from Phase 15 PRECOMMIT."""
import json,numpy as np
from pathlib import Path
O=Path('/mnt/data/isiXhosa_phase15_source_grounded_search');Q=[json.loads(s) for s in open(O/'paragraph_queries.jsonl')]
S=np.load(O/'paragraph_scores.npz')['raw'];C=np.load(O/'compact_char_scores.npz')['scores'];N=S.shape[1];M=len(Q)
a=np.argsort(-S,axis=1,kind='stable');b=np.argsort(-C,axis=1,kind='stable');r=np.empty_like(a);c=np.empty_like(b)
for j in range(len(a)):
 r[j,a[j]]=np.arange(1,N+1);c[j,b[j]]=np.arange(1,N+1)
R=1/(60+r)+1/(60+c)
idx={}
import pickle
D=pickle.load(open('/mnt/data/isiXhosa_three_track_pilot/corpus_cache.pkl','rb'))
idx={d['id']:i for i,d in enumerate(D)};out=[];summary={}
for kind,start in [('clean',0),('noisy',M)]:
 for i,q in enumerate(Q):
  z=R[i+start];order=np.argsort(-z,kind='stable');rank=np.empty(N,dtype=int);rank[order]=np.arange(1,N+1)
  gold=idx[q['page']];anygold=np.array([idx[x] for x in q['match_pages']])
  out.append({'qid':q['id'],'kind':kind,'issue':q['issue'],'source':q['source'],'lang':q['lang'],
              'exact1':int(rank[gold]==1),'any1':int(rank[anygold].min()==1),'exact5':int(rank[gold]<=5),'any5':int(rank[anygold].min()<=5)})
 A=[x for x in out if x['kind']==kind];summary[kind]={k:round(float(np.mean([x[k] for x in A])),6) for k in ('exact1','any1','exact5','any5')}
(O/'rrf_diagnostic_summary.json').write_text(json.dumps(summary,indent=2));print(summary)
