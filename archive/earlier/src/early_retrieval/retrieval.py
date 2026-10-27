"""Three fixed search baselines and language-filter controls for source-anchored page retrieval."""
import json, pickle, time, numpy as np, scipy.sparse as sp
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
OUT=Path('/mnt/data/isiXhosa_three_track_pilot')
with open(OUT/'corpus_cache.pkl','rb') as f:pages=pickle.load(f)
qs=[json.loads(l) for l in (OUT/'queries.jsonl').open()]
IDs=[r['id'] for r in pages]; IDX={x:i for i,x in enumerate(IDs)}
raw=[r['raw'].lower() for r in pages]
xh=[r['xh'].lower() for r in pages]
raw_split=[(r['xh']+'\n'+r['en']).lower() for r in pages]
clean=[q['query'] for q in qs]; noisy=[q['noisy_query'] for q in qs]; queries=clean+noisy
N=len(pages)
t0=time.time()
def bmranking(docs, name):
 print('Start BM25',name,'elapsed',round(time.time()-t0,1),flush=True)
 vec=CountVectorizer(lowercase=False,token_pattern=r'(?u)\b[^\W\d_]+\b',min_df=2,max_features=130000,dtype=np.float32)
 X=vec.fit_transform(docs).tocsr()
 if X.shape[1]<1000:raise RuntimeError('bad vocab')
 dl=np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
 avdl=max(float(dl.mean()),1)
 df=np.asarray((X>0).sum(axis=0)).ravel().astype(np.float32)
 idf=np.log(1+(N-df+0.5)/(df+0.5))
 X=X.tocoo()
 den=X.data+1.2*(0.25+0.75*dl[X.row]/avdl)
 data=(X.data*2.2/den)*idf[X.col]
 X=sp.csr_matrix((data,(X.row,X.col)),shape=X.shape,dtype=np.float32)
 print('BM25 matrix',name,X.shape,X.nnz,'elapsed',round(time.time()-t0,1),flush=True)
 Q=vec.transform(queries);Q.data[:]=1
 S=(Q@X.T).toarray().astype(np.float32)
 return S
word_raw=bmranking(raw,'raw')
np.savez_compressed(OUT/'intermediate_word_raw.npz',score=word_raw)
word_xh=bmranking(xh,'xh')
np.savez_compressed(OUT/'intermediate_word_xh.npz',score=word_xh)
word_split=bmranking(raw_split,'split_bilingual')
np.savez_compressed(OUT/'intermediate_word_split.npz',score=word_split)
print('Start char TFIDF elapsed',round(time.time()-t0,1),flush=True)
cv=TfidfVectorizer(analyzer='char',ngram_range=(3,4),min_df=4,max_features=85000,sublinear_tf=True,dtype=np.float32)
C=cv.fit_transform(raw)
print('char matrix',C.shape,C.nnz,'elapsed',round(time.time()-t0,1),flush=True)
CQ=cv.transform(queries)
char=(CQ@C.T).toarray().astype(np.float32)
np.savez_compressed(OUT/'intermediate_char.npz',score=char)
def sorted_indices(S):return np.argsort(-S,axis=1,kind='stable')
ra=sorted_indices(word_raw);rb=sorted_indices(char)
rrfa=np.empty(ra.shape,dtype=np.int32);rrfa[np.arange(len(ra))[:,None],ra]=np.arange(N)[None,:]
rrfb=np.empty(rb.shape,dtype=np.int32);rrfb[np.arange(len(rb))[:,None],rb]=np.arange(N)[None,:]
rrf=1/(60.+1.+rrfa)+1/(60.+1.+rrfb)
rankings={'BM25_raw':ra,'CHAR_raw':rb,'RRF_raw':sorted_indices(rrf),'BM25_Xh_only':sorted_indices(word_xh),'BM25_split_bilingual':sorted_indices(word_split)}
rmaj_xh=np.array([r['xh_pct']>=50 for r in pages]);rmaj_en=np.array([r['en_pct']>=50 for r in pages])
def filtered_rank(ranks,allowed):
 result=np.empty_like(ranks)
 for i,row in enumerate(ranks):
  mask=allowed[row];result[i]=np.concatenate((row[mask],row[~mask]))
 return result
rankings['BM25_majority_Xh']=filtered_rank(ra,rmaj_xh)
rankings['BM25_majority_En']=filtered_rank(ra,rmaj_en)
records=[]
for method,ranks in rankings.items():
 for j,q in enumerate(qs):
  gold=IDX[q['page']]
  for typ,i in [('clean',j),('noisy',j+len(qs))]:
   rank=np.where(ranks[i]==gold)[0][0]+1
   if method=='BM25_majority_Xh' and not rmaj_xh[gold]:rank=N+1
   if method=='BM25_majority_En' and not rmaj_en[gold]:rank=N+1
   records.append(dict(method=method,kind=typ,query_id=q['id'],lang=q['lang'],source=q['source'],issue=q['issue'],gold=q['page'],rank=int(rank),
                       r1=int(rank<=1),r5=int(rank<=5),rr=1/rank if rank<=N else 0))
with (OUT/'retrieval_results.jsonl').open('w') as f:
 for r in records:f.write(json.dumps(r)+'\n')
summary={}
for meth in rankings:
 summary[meth]={}
 for kind in ('clean','noisy'):
  for src in ('IMVO','ISIG'):
   for lang in ('xh','en'):
    z=[r for r in records if r['method']==meth and r['kind']==kind and r['source']==src and r['lang']==lang]
    if z:summary[meth][f'{kind}_{src}_{lang}']={'n':len(z),'R1':sum(r['r1'] for r in z)/len(z),'R5':sum(r['r5'] for r in z)/len(z),'MRR':sum(r['rr'] for r in z)/len(z)}
(OUT/'retrieval_summary.json').write_text(json.dumps(summary,indent=2))
print('done',round(time.time()-t0,1),'seconds',flush=True)
for m in ('BM25_raw','CHAR_raw','RRF_raw','BM25_Xh_only','BM25_split_bilingual','BM25_majority_Xh'):
 print(m, {k:(round(v['R1'],3),round(v['R5'],3)) for k,v in summary[m].items()},flush=True)
