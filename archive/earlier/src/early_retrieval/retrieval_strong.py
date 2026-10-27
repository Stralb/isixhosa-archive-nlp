"""Unrestricted BM25 and evidence-preserving vs majority-only language metadata filters."""
from pathlib import Path
import gc,json,pickle,time,numpy as np,scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
O=Path('/mnt/data/isiXhosa_three_track_pilot')
with (O/'corpus_cache.pkl').open('rb') as f:docs=pickle.load(f)
q=[json.loads(s) for s in (O/'queries.jsonl').open()]
queries=[r['query'] for r in q]+[r['noisy_query'] for r in q]
N=len(docs); M=len(q); id2n={r['id']:j for j,r in enumerate(docs)}
t0=time.time()
def full_bm25(view, label):
 print('FIT',label,round(time.time()-t0,1),flush=True)
 vector=CountVectorizer(lowercase=True,token_pattern=r'(?u)\b[^\W\d_]+\b',dtype=np.float32)
 texts=[(r[view] if view!='split' else r['xh']+'\n'+r['en']) for r in docs]
 X=vector.fit_transform(texts).tocsr()
 print('vocab matrix',label,X.shape,'nnz',X.nnz,'elapsed',round(time.time()-t0,1),flush=True)
 dl=np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
 idf=np.log1p((N-np.asarray((X>0).sum(axis=0)).ravel()+.5)/(np.asarray((X>0).sum(axis=0)).ravel()+.5))
 av=max(dl.mean(),1.0)
 X=X.tocoo(); data=(X.data*2.2/(X.data+1.2*(.25+.75*dl[X.row]/av)))*idf[X.col]
 weighted=sp.csr_matrix((data,(X.row,X.col)),shape=X.shape,dtype=np.float32)
 Q=vector.transform(queries);Q.data[:]=1
 scores=(Q@weighted.T).toarray().astype(np.float32)
 del X,weighted,Q,vector,texts;gc.collect()
 np.savez_compressed(O/f'strong_{label}_scores.npz',score=scores)
 return scores

raw=full_bm25('raw','raw')
xh=full_bm25('xh','xh')
split=full_bm25('split','split')
char=np.load(O/'intermediate_char.npz')['score']

def ranks(scores):return np.argsort(-scores,axis=1,kind='stable')
ra=ranks(raw);ca=ranks(char)
rrfa=np.empty_like(ra,dtype=np.int32);rrfb=np.empty_like(ca,dtype=np.int32)
rrfa[np.arange(2*M)[:,None],ra]=np.arange(N)[None,:]
rrfb[np.arange(2*M)[:,None],ca]=np.arange(N)[None,:]
rrf=ranks(1/(60.+1.+rrfa)+1/(60.+1.+rrfb))
ranklists={'BM25_full_raw':ra,'BM25_full_xh_only':ranks(xh),'BM25_full_split_bilingual':ranks(split),'Char_truncated_raw':ca,'RRF_full_BM25_plus_char':rrf}
for lang in ('xh','en'):
 if lang=='xh':
  major=np.array([r['xh_pct']>=50 for r in docs]); tags=np.array([int(r.get('tracker_xh_words') or 0)>=50 for r in docs]);
 else:
  major=np.array([r['en_pct']>=50 for r in docs]); tags=np.array([int(r.get('tracker_en_words') or 0)>=50 for r in docs]);
 for tag,allow in [('majority',major),('multitag_50words',tags)]:
  rank=np.empty_like(ra)
  for i,row in enumerate(ra):rank[i]=np.concatenate((row[allow[row]],row[~allow[row]]))
  ranklists[f'BM25_filter_{lang}_{tag}']=(rank,allow)
print('ranking all',round(time.time()-t0,1),flush=True)
res=[]
for method,obj in ranklists.items():
 if isinstance(obj,tuple):rank,allowed=obj
 else: rank,allowed=obj,None
 for i,record in enumerate(q):
  gold=id2n[record['page']]
  for kind,j in [('clean',i),('noisy',i+M)]:
   # Query should only match first rank if its page is indexed and there is lexical evidence.
   idx=np.where(rank[j]==gold)[0][0]
   position=(idx+1) if (allowed is None or allowed[gold]) else N+1
   if method=='BM25_full_xh_only' and float(xh[j,gold])<=0:position=N+1
   if method.startswith('BM25_full') and 'raw' in method and float(raw[j,gold])<=0:position=N+1
   res.append(dict(method=method,kind=kind,source=record['source'],lang=record['lang'],qid=record['id'],page=record['page'],issue=record['issue'],
        rank=int(position),r1=int(position==1),r5=int(position<=5),recip=float(1/position) if position<=N else 0,
        page_present=bool(allowed is None or allowed[gold])))
with (O/'retrieval_strong_results.jsonl').open('w') as f:
 for o in res:f.write(json.dumps(o)+'\n')
summary={}
for method in ranklists:
 summary[method]={}
 for kind in ('clean','noisy'):
  for source in ('IMVO','ISIG'):
   for lang in ('xh','en'):
    sample=[r for r in res if r['method']==method and r['kind']==kind and r['source']==source and r['lang']==lang]
    if sample:
     summary[method][f'{kind}_{source}_{lang}']={
       'n':len(sample),'R1':float(np.mean([x['r1'] for x in sample])),'R5':float(np.mean([x['r5'] for x in sample])),
       'MRR':float(np.mean([x['recip'] for x in sample])),
       'page_coverage':float(np.mean([x['page_present'] for x in sample]))}
(O/'retrieval_strong_summary.json').write_text(json.dumps(summary,indent=2))
for method in ranklists:
 print(method,{k:(round(v['R1'],3),round(v['R5'],3)) for k,v in summary[method].items()},flush=True)
print('Done',round(time.time()-t0,1),flush=True)
