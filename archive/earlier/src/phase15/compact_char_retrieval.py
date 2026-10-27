"""Compact character trigram baseline after full char analysis timeout, described in EXECUTION_NOTE."""
from pathlib import Path
import pickle,json,time,collections,numpy as np,regex
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
O=Path('/mnt/data/isiXhosa_phase15_source_grounded_search');P=Path('/mnt/data/isiXhosa_three_track_pilot');t0=time.time()
D=pickle.load(open(P/'corpus_cache.pkl','rb'));Q=[json.loads(x) for x in open(O/'paragraph_queries.jsonl')];qs=[q['query'] for q in Q]+[q['noisy_query'] for q in Q]
W=regex.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
@lru_cache(maxsize=200000)
def cg(w):
 s='^'+w.lower()+'$'
 return tuple(s[i:i+3] for i in range(len(s)-2))
def analyzer(text):
 for w in W.findall(text):
  yield from cg(w)
print('FIT compact word-internal char 3gram tfidf',flush=True)
v=TfidfVectorizer(analyzer=analyzer,min_df=2,max_features=75000,sublinear_tf=True,dtype=np.float32)
X=v.fit_transform([d['raw'] for d in D]);print('X fit',X.shape,X.nnz,'seconds',round(time.time()-t0,1),flush=True)
Y=v.transform(qs);s=(Y@X.T).toarray().astype(np.float32)
print('Scored',s.shape,'seconds',round(time.time()-t0,1),flush=True)
np.savez_compressed(O/'compact_char_scores.npz',scores=s)
idx={d['id']:i for i,d in enumerate(D)};N=len(D);M=len(Q);out=[];summ={}
for typ,offset in [('clean',0),('noisy',M)]:
 for i,q in enumerate(Q):
  row=np.argsort(-s[i+offset],kind='stable');r=np.empty(N,dtype=np.int32);r[row]=np.arange(1,N+1)
  gold=idx[q['page']];goods=np.array([idx[x] for x in q['match_pages']])
  er=int(r[gold]) if s[i+offset,gold]>0 else N+1
  valid=goods[s[i+offset,goods]>0];ar=int(r[valid].min()) if len(valid) else N+1
  out.append(dict(method='compact_char3_tfidf',kind=typ,qid=q['id'],source=q['source'],lang=q['lang'],issue=q['issue'],page=q['page'],page_num=q['source_page_num'],
                  multi=int(q['num_exact_match_pages']>1),exact1=int(er==1),any1=int(ar==1),exact5=int(er<=5),any5=int(ar<=5)))
 a=[x for x in out if x['kind']==typ]
 summ[typ]={k:round(float(np.mean([x[k] for x in a])),6) for k in ('exact1','any1','exact5','any5')}
 summ[typ]['n']=len(a)
with (O/'compact_char_results.jsonl').open('w') as f:
 for x in out:f.write(json.dumps(x)+'\n')
(O/'compact_char_summary.json').write_text(json.dumps(summ,indent=2))
print('SUMMARY',summ,'sec',round(time.time()-t0,1),flush=True)
