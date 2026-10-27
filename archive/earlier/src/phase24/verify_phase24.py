"""Independent Phase24 integrity verification. Recalculates counts, page-exact gold, all score/ranks and gates."""
from __future__ import annotations
import collections,hashlib,json,re,time,math
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from rapidfuzz import fuzz
import regex
R=Path(__file__).resolve().parents[1];O=R/'phase24';start=time.time()
sh=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for f,h in [('PRECOMMIT.md','PRECOMMIT_SHA256.txt'),('sample_frozen.jsonl','SAMPLE_SHA256.txt'),('queries_gold_frozen.jsonl','GOLD_SHA256.txt')]:
 assert sh(O/f)==(O/h).read_text().split()[0],f'Hash mismatch {f}'
D=[json.loads(x) for x in (R/'pages.jsonl').open()];Q=[json.loads(x) for x in (O/'queries_gold_frozen.jsonl').open()]
P=[json.loads(x) for x in (O/'per_query.jsonl').open()]
assert len(D)==3608 and len(Q)==len(P)==1436
ID={d['id']:j for j,d in enumerate(D)}; issue={d['id']:d['issue'] for d in D}
W=regex.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
def tok(s):return tuple(w.lower() for w in W.findall(s))
def eligible(z):
 if len(z)!=13 or len(set(z))<8:return False
 return len({t for t in z if len(t)>=4 and t.isalpha()})>=6
# Validate each q and orig anchor within exact paragraph and reconstruct cross-archive gold.
unique={tuple(q['anchor_tokens']) for q in Q}
first=collections.defaultdict(set)
for z in unique:first[z[0]].add(z)
seen=collections.defaultdict(set)
for p in D:
 for s in re.split(r'\n\s*\n',p['raw']):
  t=tok(s)
  for k in range(max(0,len(t)-12)):
   if t[k] not in first:continue
   z=t[k:k+13]
   if z in first[z[0]]:seen[z].add(p['id'])
print('independent exact-match archive scan',round(time.time()-start,1),flush=True)
for q,p in zip(Q,P):
 assert q['qid']==p['qid']
 assert q['issue']==issue[q['page_id']]
 z=tuple(q['anchor_tokens'])
 assert eligible(z)
 para=tok(re.split(r'\n\s*\n',D[ID[q['page_id']]]['raw'])[q['paragraph_index']])
 assert 50<=len(para)<=350
 assert z==para[q['anchor_start']:q['anchor_start']+13]
 assert sorted(seen[z])==q['exact_gold_pages']
 assert q['exact_gold_issue_count']==len({issue[p] for p in seen[z]})
 assert q['cross_issue_repeat']==(q['exact_gold_issue_count']>=2)
 assert q['within_issue_multipage_repeat']==(len(seen[z])>1 and q['exact_gold_issue_count']==1)
 query=tuple(q['query'].split());assert len(query)==6
 assert all(query[i:i+3] not in {z[j:j+3] for j in range(11)} for i in range(4))
 assert q['query']== ' '.join(sorted(sorted({t for t in z if len(t)>=4 and t.isalpha()},key=lambda x:(-len(x),x))[:6]))
# Independently regenerate BM25 scores and all top20 rankings
cv=CountVectorizer(lowercase=True,token_pattern=r'(?u)\b[^\W\d_]+\b',dtype=np.float32)
X=cv.fit_transform([p['raw'] for p in D]).tocsr()
lengths=np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
freq=np.asarray((X>0).sum(axis=0)).ravel()
idf=np.log1p((len(D)-freq+0.5)/(freq+0.5))
a=float(lengths.mean())
coo=X.tocoo()
coef=coo.data*2.2/(coo.data+1.2*(.25+.75*lengths[coo.row]/a))*idf[coo.col]
M=sp.csr_matrix((coef,(coo.row,coo.col)),shape=X.shape,dtype=np.float32)
Y=cv.transform([q['query'] for q in Q]);Y.data[:]=1
s=(Y@M.T).toarray().astype(np.float32)
original=np.load(O/'phase24_bm25_scores.npz')['raw_bm25']
assert np.array_equal(s,original),f'SCORE MISMATCH {np.max(np.abs(s-original))}'
idx=np.argsort(-s,axis=1,kind='stable')
norm=lambda st:re.sub('[^a-z0-9]','',st.lower())
N=[norm(p['raw']) for p in D]
qnorm=[norm(q['query']) for q in Q]
for i,(q,r) in enumerate(zip(Q,P)):
 pool=[int(j) for j in idx[i] if s[i,j]>0][:20]
 sim={j:float(fuzz.partial_ratio(qnorm[i],N[j])) for j in pool}
 rerank=sorted(pool,key=lambda j:(-sim[j],-float(s[i,j]),j))
 orig=ID[q['page_id']];gold={ID[pg] for pg in q['exact_gold_pages']}
 for m,ids in [('BM25',pool),('BM25_FUZZY20',rerank)]:
  item=r['methods'][m]
  assert [D[j]['id'] for j in ids]==item['ranked_top20']
  assert [D[j]['id'] for j in pool]==item['candidate_pool']
  assert item['origin_top1']==int(bool(ids) and ids[0]==orig)
  assert item['evidence_top1']==int(bool(ids) and ids[0] in gold)
  assert item['origin_recall20']==int(orig in pool)
  assert item['evidence_recall20']==int(any(j in gold for j in pool))
 if (i+1)%500==0:print('verified rankings',i+1,'s',round(time.time()-start,1),flush=True)
# Recalculate macro statistics and all three gates without calling analysis script.
S=['IMVO','ISIG'];F=['A','B'];look={(r['issue'],r['frame']):r for r in P}
rs=json.loads((O/'phase24_results.json').read_text())
for fr in F:
 for metric,fn in {
 'cross_issue_repeat':lambda v:int(v['cross_issue_repeat']),
 'bm25_evidence_minus_origin':lambda v:v['methods']['BM25']['evidence_top1']-v['methods']['BM25']['origin_top1'],
 'model_comparison_diff_in_diff':lambda v:(v['methods']['BM25_FUZZY20']['evidence_top1']-v['methods']['BM25']['evidence_top1'])-(v['methods']['BM25_FUZZY20']['origin_top1']-v['methods']['BM25']['origin_top1']),
 }.items():
  means=[]
  for source in S:
   vals=[fn(look[(r['issue'],fr)]) for r in P if r['source']==source and r['frame']==fr]
   assert len(vals)==(517 if source=='IMVO' else 201)
   means.append(float(np.mean(vals)))
  observed=rs['metrics'][fr][metric]
  assert abs(.5*sum(means)-observed['macro'])<1e-12
  assert all(abs(means[i]-observed['by_source'][source])<1e-12 for i,source in enumerate(S))
# independent bootstrap verification: regenerate same RNG source-level arrays and CIs
rng=np.random.default_rng(2026102424)
issue_list={s:sorted({r['issue'] for r in P if r['source']==s}) for s in S}
ind={s:rng.integers(0,len(issue_list[s]),size=(10000,len(issue_list[s]))) for s in S}
for fr in F:
 for metric,fn in {
 'cross_issue_repeat':lambda v:int(v['cross_issue_repeat']),
 'bm25_evidence_minus_origin':lambda v:v['methods']['BM25']['evidence_top1']-v['methods']['BM25']['origin_top1'],
 'model_comparison_diff_in_diff':lambda v:(v['methods']['BM25_FUZZY20']['evidence_top1']-v['methods']['BM25']['evidence_top1'])-(v['methods']['BM25_FUZZY20']['origin_top1']-v['methods']['BM25']['origin_top1'])
 }.items():
  bootstrap=[]
  for source in S:
   vals=np.array([fn(look[(iss,fr)]) for iss in issue_list[source]],float)
   bootstrap.append(vals[ind[source]].mean(axis=1))
  ci=np.quantile(.5*(bootstrap[0]+bootstrap[1]),[.025,.975])
  assert np.allclose(ci,rs['metrics'][fr][metric]['ci95'])
assert not any(v['passes_all_gates'] for v in rs['hypotheses'].values())
print('INDEPENDENT FULL VERIFICATION PASSED',json.dumps({
 'n_orig_pages':3608,'n_pairs':1436,'full_bm25_max_absolute_diff':float(np.max(np.abs(s-original))),
 'all_query_gold_exact':True,'all_query_rankings_exact':True,
 'bootstrap_CIs_exact':True,'gates_recalculated':True,
 'seconds':round(time.time()-start,2)},indent=2),flush=True)
