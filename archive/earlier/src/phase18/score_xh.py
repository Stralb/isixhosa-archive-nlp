from pathlib import Path
import pickle,json,numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as sp
P=Path('/mnt/data/isiXhosa_phase18_evidence_routing')
D=pickle.load(open('/mnt/data/isiXhosa_three_track_pilot/corpus_cache.pkl','rb'))
q=[json.loads(x) for x in open(P/'balanced_xh_queries.jsonl')]
N=len(D);M=len(q)
scores={}
for field in ('raw','en','xh'):
 print('START',field,flush=True)
 v=CountVectorizer(lowercase=True,token_pattern=r'(?u)\b[^\W\d_]+\b',dtype=np.float32)
 X=v.fit_transform([d[field] for d in D]).tocsr()
 lengths=np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
 df=np.asarray((X>0).sum(axis=0)).ravel()
 idf=np.log1p((N-df+.5)/(df+.5))
 mu=max(float(lengths.mean()),1.)
 c=X.tocoo()
 wt=c.data*2.2/(c.data+1.2*(.25+.75*lengths[c.row]/mu))*idf[c.col]
 weights=sp.csr_matrix((wt,(c.row,c.col)),shape=c.shape,dtype=np.float32)
 T=v.transform([x['text'] for x in q]);T.data[:]=1
 scores[field]=(T@weights.T).toarray().astype(np.float32)
 print('END',field,'max score',np.median(scores[field].max(axis=1)),flush=True)
 del v,X,c,T,weights
np.savez_compressed(P/'balanced_xh_bm25_scores.npz',**scores)
print('COMPLETE',flush=True)
