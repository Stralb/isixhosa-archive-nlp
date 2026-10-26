from pathlib import Path
from collections import defaultdict
import pickle,json,time,re, numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from rapidfuzz.fuzz import partial_ratio
P=Path('/mnt/data/isiXhosa_phase17_deeper_questions')
O=Path('/mnt/data/isiXhosa_phase15_source_grounded_search')
D=pickle.load(open('/mnt/data/isiXhosa_three_track_pilot/corpus_cache.pkl','rb'))
Q=[json.loads(x) for x in open(O/'paragraph_queries.jsonl')]
S=np.load(O/'paragraph_scores.npz')['raw']; M=len(Q); N=len(D)
index={d['id']:i for i,d in enumerate(D)}
assert S.shape==(2*M,N)
# Same BM25 and lexical vocabulary as Phase 15, raw-only for short query comparison.
def bm25_short(docs,qtexts):
 v=CountVectorizer(lowercase=True,token_pattern=r'(?u)\b[^\W\d_]+\b',dtype=np.float32)
 X=v.fit_transform(docs).tocsr();lens=np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
 df=np.asarray((X>0).sum(axis=0)).ravel();idf=np.log1p((N-df+.5)/(df+.5));mu=max(float(lens.mean()),1)
 co=X.tocoo(); w=co.data*2.2/(co.data+1.2*(.25+.75*lens[co.row]/mu))*idf[co.col]
 W=sp.csr_matrix((w,(co.row,co.col)),shape=co.shape,dtype=np.float32)
 T=v.transform(qtexts);T.data[:]=1
 return (T@W.T).toarray().astype(np.float32)
short=[' '.join(q['query'].split()[3:7]) for q in Q]
T=time.time(); short_scores=bm25_short([d['raw'] for d in D],short)
print('short BM25',short_scores.shape,'elapsed',round(time.time()-T,1),flush=True)
# Normalize only letters/digits, no word boundaries, to handle synthetic split/join in queries.
sub=re.compile(r'[^a-z0-9]')
page_norm=[sub.sub('',d['raw'].lower()) for d in D]
result=[]
methods=[('bm25_10_clean',S[:M]),('bm25_10_noisy',S[M:]),('bm25_4_clean',short_scores)]
for method,scores in methods:
 for i,q in enumerate(Q):
  ranks=np.argsort(-scores[i],kind='stable')
  goods={index[x] for x in q['match_pages']}; src=index[q['page']]
  assert src in goods
  top1=int(ranks[0] in goods and scores[i,ranks[0]]>0)
  top5=int(any(k in goods and scores[i,k]>0 for k in ranks[:5]))
  top20=int(any(k in goods and scores[i,k]>0 for k in ranks[:20]))
  result.append(dict(method=method,qid=q['id'],source=q['source'],lang=q['lang'],issue=q['issue'],page_num=q['source_page_num'],multi=int(q['num_exact_match_pages']>1),any1=top1,any5=top5,top20=top20,exact1=int(ranks[0]==src and scores[i,ranks[0]]>0)))
 print('Completed',method,flush=True)
# Fuzzy reranking is done *only* inside raw BM25 top-20; no reranking when no positive lexical evidence.
for method,rows,queries in [('fuzzy_10_clean',S[:M],[q['query'] for q in Q]),('fuzzy_10_noisy',S[M:],[q['noisy_query'] for q in Q])]:
 for i,q in enumerate(Q):
  candidates=np.argsort(-rows[i],kind='stable')[:20]
  candidates=[int(j) for j in candidates if rows[i,j]>0]
  query=sub.sub('',queries[i].lower())
  z=[(float(partial_ratio(query,page_norm[j])),float(rows[i,j]),j) for j in candidates]
  z.sort(key=lambda t:(-t[0],-t[1],t[2]))
  ranks=[x[2] for x in z]
  goods={index[x] for x in q['match_pages']};src=index[q['page']]
  result.append(dict(method=method,qid=q['id'],source=q['source'],lang=q['lang'],issue=q['issue'],page_num=q['source_page_num'],multi=int(q['num_exact_match_pages']>1),any1=int(bool(ranks) and ranks[0] in goods),any5=int(any(j in goods for j in ranks[:5])),top20=int(any(j in goods for j in ranks)),exact1=int(bool(ranks) and ranks[0]==src)))
 print('Completed',method,'elapsed',round(time.time()-T,1),flush=True)
summ={}
for meth in sorted(set(x['method'] for x in result)):
 a=[x for x in result if x['method']==meth];summ[meth]={}
 for name,group in [('all',a),('unique_nonfront',[x for x in a if not x['multi'] and x['page_num']>1]),('IMVO_xh',[x for x in a if x['source']=='IMVO' and x['lang']=='xh']),('IMVO_en',[x for x in a if x['source']=='IMVO' and x['lang']=='en']),('ISIG_xh',[x for x in a if x['source']=='ISIG' and x['lang']=='xh']),('ISIG_en',[x for x in a if x['source']=='ISIG' and x['lang']=='en'])]:
  if group: summ[meth][name]={'n':len(group),**{k:round(float(np.mean([x[k] for x in group])),5) for k in ('any1','any5','top20','exact1')}}
print(json.dumps(summ,indent=2),flush=True)
with (P/'retrieval_probe_results.jsonl').open('w') as f:
 for r in result:f.write(json.dumps(r)+'\n')
(P/'retrieval_probe_summary.json').write_text(json.dumps(summ,indent=2))
# Paired issue-group bootstrap: any1 main: fuzzy noisy vs BM25 noisy; BM25 short vs original clean.
rng=np.random.default_rng(20261023)
bases={m:{x['qid']:x for x in result if x['method']==m} for m in summ}
q_by_id={q['id']:q for q in Q}
strata={source:sorted({q['issue'] for q in Q if q['source']==source}) for source in ('IMVO','ISIG')}
ids_by_issue=defaultdict(list)
for q in Q: ids_by_issue[q['issue']].append(q['id'])
def bootstrap(a,b,n=5000):
 point=float(np.mean([bases[a][q['id']]['any1']-bases[b][q['id']]['any1'] for q in Q])); samples=[]
 for z in range(n):
  ids=[]
  for s,issues in strata.items():
   choices=rng.choice(issues,size=len(issues),replace=True)
   ids.extend(qid for iss in choices for qid in ids_by_issue[iss])
  samples.append(float(np.mean([bases[a][k]['any1']-bases[b][k]['any1'] for k in ids])))
 return {'point':point,'ci95':[float(x) for x in np.quantile(samples,[.025,.975])], 'issues':sum(map(len,strata.values()))}
(P/'retrieval_probe_uncertainty.json').write_text(json.dumps({'fuzzy_noisy_vs_raw':bootstrap('fuzzy_10_noisy','bm25_10_noisy'),'four_words_vs_ten':bootstrap('bm25_4_clean','bm25_10_clean')},indent=2))
print('TOTAL seconds',round(time.time()-T,1),flush=True)
