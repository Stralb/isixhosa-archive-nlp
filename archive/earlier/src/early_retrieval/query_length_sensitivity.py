"""Query length 5/10/20 source ambiguity and unrestricted raw BM25 search."""
from pathlib import Path
from collections import defaultdict, Counter
import pickle,json,regex as rx,time, numpy as np, scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
O=Path('/mnt/data/isiXhosa_three_track_pilot');t0=time.time()
with (O/'corpus_cache.pkl').open('rb') as f:pages=pickle.load(f)
qs=[json.loads(s) for s in (O/'query_repaired.jsonl').open()];docidx={d['id']:j for j,d in enumerate(pages)}
WORD=rx.compile(r'(?V1)\p{L}+(?:[\'’]\p{L}+)*')
def words(s):return [w.lower() for w in WORD.findall(s)]
T={r['id']:words(r['raw']) for r in pages}
Q=[]
for q in qs:
 tok=T[q['page']]; needle=words(q['query'])
 start=next((i for i in range(len(tok)-9) if tok[i:i+10]==needle),None)
 assert start is not None,q['id']
 for n in (5,10,20):
  if n==5: new=tok[start+2:start+7]
  elif n==10:new=needle
  else:
   si=max(0,min(start-5,len(tok)-20));new=tok[si:si+20]
   assert len(new)==20
  Q.append({'qid':q['id'],'source':q['source'],'lang':q['lang'],'issue':q['issue'],'page':q['page'],
            'nwords':n,'query':' '.join(new),'tokens':new})
B=1099511628211;MASK=(1<<64)-1

def hashes(seq,n):
 if len(seq)<n:return []
 a=[hash(w)&MASK for w in seq];power=pow(B,n,1<<64)
 h=0;out=[]
 for i,x in enumerate(a):
  h=(h*B+x)&MASK
  if i>=n:h=(h-a[i-n]*power)&MASK
  if i>=n-1:out.append(h)
 return out

def h1(seq):
 a=0
 for w in seq:a=(a*B+(hash(w)&MASK))&MASK
 return a
lookup={n:defaultdict(list) for n in (5,10,20)}
for i,q in enumerate(Q):lookup[q['nwords']][h1(q['tokens'])].append(i)
matched=[set() for q in Q]
for pi,doc in enumerate(pages):
 seq=T[doc['id']]
 for n in (5,10,20):
  hset=set(hashes(seq,n))
  for h in hset:
   for i in lookup[n].get(h,[]):matched[i].add(pi)
 if (pi+1)%1200==0:print('ngram pass',pi+1,round(time.time()-t0,1),flush=True)
for i,q in enumerate(Q):
 assert docidx[q['page']] in matched[i]
 q['n_matching_pages']=len(matched[i]);q['n_matching_issues']=len({pages[j]['issue'] for j in matched[i]})
# Strong BM25 exact full vocabulary
v=CountVectorizer(lowercase=True,token_pattern=r'(?u)\b[^\W\d_]+\b',dtype=np.float32)
X=v.fit_transform([r['raw'] for r in pages]).tocsr();N=len(pages)
length=np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
df=np.asarray((X>0).sum(axis=0)).ravel();idf=np.log1p((N-df+.5)/(df+.5))
co=X.tocoo();data=co.data*2.2/(co.data+1.2*(.25+.75*length[co.row]/max(float(length.mean()),1)))*idf[co.col]
weighted=sp.csr_matrix((data,(co.row,co.col)),shape=co.shape,dtype=np.float32)
qq=v.transform([z['query'] for z in Q]);qq.data[:]=1
S=(qq@weighted.T).toarray()
R=np.argsort(-S,axis=1,kind='stable')
for i,q in enumerate(Q):
 ranks=np.empty(N,dtype=np.int32);ranks[R[i]]=np.arange(1,N+1)
 q['rank_exact']=int(ranks[docidx[q['page']]]);q['rank_any']=int(min(ranks[list(matched[i])]))
 for k in ('tokens',):del q[k]
with (O/'query_length_results.jsonl').open('w') as f:
 for q in Q:f.write(json.dumps(q)+'\n')
summary={}
for n in (5,10,20):
 z=[r for r in Q if r['nwords']==n]
 summary[n]={}
 for group,subset in [('all',z)]+[(src+'_'+lang,[q for q in z if q['source']==src and q['lang']==lang]) for src in ('IMVO','ISIG') for lang in ('xh','en')]:
  if not subset:continue
  summary[n][group]={'n_queries':len(subset),'multiple_page_frac':float(np.mean([r['n_matching_pages']>1 for r in subset])),
    'multiple_issue_frac':float(np.mean([r['n_matching_issues']>1 for r in subset])),
    'source_R1':float(np.mean([r['rank_exact']==1 for r in subset])),
    'any_R1':float(np.mean([r['rank_any']==1 for r in subset])),
    'source_R5':float(np.mean([r['rank_exact']<=5 for r in subset])),
    'any_R5':float(np.mean([r['rank_any']<=5 for r in subset])),
    'median_match_pages':float(np.median([r['n_matching_pages'] for r in subset])),
    'max_match_pages':int(max(r['n_matching_pages'] for r in subset))}
(O/'query_length_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2),flush=True)
print('DONE time',round(time.time()-t0,1),flush=True)
