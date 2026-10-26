"""Run the prospectively frozen Phase21 benchmark, no retrieval-method retuning."""
from __future__ import annotations
import collections, hashlib, json, math, re, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from rapidfuzz.fuzz import partial_ratio
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'phase21'
D=[json.loads(x) for x in (ROOT/'pages.jsonl').open()]
Q=[json.loads(x) for x in (OUT/'queries_frozen.jsonl').open()]
assert len(D)==3608 and len(Q)==300
assert hashlib.sha256((OUT/'queries_frozen.jsonl').read_bytes()).hexdigest()==json.loads((OUT/'construction_summary.json').read_text())['query_file_sha256']
assert hashlib.sha256((OUT/'PRECOMMIT.md').read_bytes()).hexdigest()==(OUT/'PRECOMMIT_SHA256.txt').read_text().split()[0]
index={p['id']:i for i,p in enumerate(D)}
forms=('keyword','clean','edit1','edit2_boundary','edit3_boundary')
methods=('word20','word20_fuzzy','char20','char20_fuzzy','hybrid15w5c','hybrid15w5c_fuzzy','xh20_fuzzy')
print('START Phase21',len(D),'archive pages',len(Q),'queries',flush=True)

# Identical Phase20 word BM25 scoring (but one query vector is binary).
n=len(D)
def bm25_index(field):
 t=time.time()
 cv=CountVectorizer(lowercase=True, token_pattern=r'(?u)\b[^\W\d_]+\b',dtype=np.float32)
 X=cv.fit_transform([d[field] for d in D]).tocsr()
 ln=np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
 df=np.asarray((X>0).sum(axis=0)).ravel()
 idf=np.log1p((n-df+.5)/(df+.5))
 avg=max(float(ln.mean()),1.)
 C=X.tocoo()
 weights=C.data*2.2/(C.data+1.2*(.25+.75*ln[C.row]/avg))*idf[C.col]
 WB=sp.csr_matrix((weights,(C.row,C.col)),shape=C.shape,dtype=np.float32)
 q=cv.transform([q['query'] for q in Q]);q.data[:]=1.
 score=(q@WB.T).toarray().astype(np.float32)
 print('BM25',field,'vocab',X.shape[1],'nnz',X.nnz,'time',round(time.time()-t,2),flush=True)
 return score, X, cv

word_score,word_counts,word_cv=bm25_index('raw')
xh_score,xh_counts,xh_cv=bm25_index('xh')
del xh_counts,xh_cv

# Word-contained, padded character trigrams.  Efficient exact sparse operation
# from the full document-word matrix and a word-type-to-character-trigram map.
# No downsampling, OOV word filter on queries, tuning or feature cap.
t0=time.time()
vocab=word_cv.get_feature_names_out()
tri_to_idx={};ci=[];val=[];ptr=[0]
for j,w in enumerate(vocab):
 s='^'+w+'$'
 grams=collections.Counter(s[i:i+3] for i in range(len(s)-2))
 for g,c in grams.items():
  col=tri_to_idx.setdefault(g,len(tri_to_idx))
  ci.append(col);val.append(c)
 ptr.append(len(ci))
 if (j+1)%150000==0:print('Char trigram map',j+1,'word features',len(tri_to_idx),'grams',flush=True)
M=sp.csr_matrix((np.array(val,dtype=np.float32),np.array(ci,dtype=np.int32),np.array(ptr,dtype=np.int32)),
                shape=(len(vocab),len(tri_to_idx)),dtype=np.float32)
del ci,val,ptr,vocab
Xgrams=(word_counts@M).tocsr();del M,word_counts
idf_transform=TfidfTransformer(norm='l2',sublinear_tf=True,smooth_idf=True)
Xgrams=idf_transform.fit_transform(Xgrams)
print('Char page grams',Xgrams.shape,'nnz',Xgrams.nnz,'elapsed',round(time.time()-t0,1),flush=True)

# Query processing uses matching lowercased alphabetic regex but DOES NOT
# discard words unseen in the index; OOV character trigrams are dropped only
# if the gram itself is absent from the index.
word_pat=re.compile(r'(?u)\b[^\W\d_]+\b')
rr=[];cc=[];vv=[]
for i,q in enumerate(Q):
 grams=collections.Counter()
 for word in word_pat.findall(q['query'].lower()):
  s='^'+word+'$';grams.update(s[k:k+3] for k in range(len(s)-2))
 for gram,c in grams.items():
  if gram in tri_to_idx:
   rr.append(i);cc.append(tri_to_idx[gram]);vv.append(c)
qraw=sp.csr_matrix((np.asarray(vv,dtype=np.float32),(rr,cc)),shape=(len(Q),len(tri_to_idx)),dtype=np.float32)
qgrams=idf_transform.transform(qraw)
char_score=(qgrams@Xgrams.T).toarray().astype(np.float32)
assert char_score.shape==(300,3608) and np.all(np.isfinite(char_score))
print('Char cosine scored',len(Q),'queries','elapsed',round(time.time()-t0,1),flush=True)
del Xgrams,word_cv,qraw,qgrams,idf_transform,tri_to_idx

# Each word/char ranked list is fully defined before any gold comparisons.
raw_rank=np.argsort(-word_score,axis=1,kind='stable')
char_rank=np.argsort(-char_score,axis=1,kind='stable')
xh_rank=np.argsort(-xh_score,axis=1,kind='stable')
raw_rpos=np.argsort(raw_rank,axis=1,kind='stable')
char_rpos=np.argsort(char_rank,axis=1,kind='stable')
np.savez_compressed(OUT/'phase21_scores.npz',word=word_score,character=char_score,xh=xh_score)

def candidate_list(i,rank,score,k):
 return [int(j) for j in rank[i] if score[i,j]>0][:k]

def hybrid_pool(i):
 rw=candidate_list(i,raw_rank,word_score,60)
 rc=candidate_list(i,char_rank,char_score,60)
 pool=rw[:15];seen=set(pool)
 for j in rc:
  if len(pool)>=20:break
  if j not in seen:pool.append(j);seen.add(j)
 for j in rw:
  if len(pool)>=20:break
  if j not in seen:pool.append(j);seen.add(j)
 if len(pool)<20:
  for j in rc:
   if len(pool)>=20:break
   if j not in seen:pool.append(j);seen.add(j)
 return pool[:20]

norm=re.compile(r'[^a-z0-9]')
page_norm=[norm.sub('',d['raw'].lower()) for d in D]
query_norm=[norm.sub('',q['query'].lower()) for q in Q]
per=[];st=time.time()
for i,q in enumerate(Q):
 w20=candidate_list(i,raw_rank,word_score,20)
 c20=candidate_list(i,char_rank,char_score,20)
 x20=candidate_list(i,xh_rank,xh_score,20)
 h20=hybrid_pool(i)
 pools={'word20':w20,'word20_fuzzy':w20,'char20':c20,'char20_fuzzy':c20,
        'hybrid15w5c':h20,'hybrid15w5c_fuzzy':h20,'xh20_fuzzy':x20}
 assert all(len(arr)<=20 for arr in pools.values())
 gold={index[x] for x in q['gold_pages']}
 origin=index[q['source_page']]
 fuzz_cache={}
 entry={'qid':q['qid'],'anchor_id':q['anchor_id'],'source':q['source'],'view':q['view'],
        'form':q['form'],'issue':q['issue'],'query':q['query'],'source_page':q['source_page'],
        'gold_pages':q['gold_pages'],'gold_count':len(gold)}
 for name,arr in pools.items():
  if name.endswith('_fuzzy'):
   for j in arr:
    if j not in fuzz_cache:fuzz_cache[j]=partial_ratio(query_norm[i],page_norm[j])
   ordered=sorted(arr,key=lambda j:(-fuzz_cache[j],-float(word_score[i,j]),j))
  elif name=='hybrid15w5c':
   ordered=sorted(arr,key=lambda j:(-1/(60+1+int(raw_rpos[i,j]))-1/(60+1+int(char_rpos[i,j])),
                      -float(word_score[i,j]),j))
  else:ordered=arr
  entry[name]={'recall20':int(bool(gold & set(arr))),
     'origin_recall20':int(origin in arr),
     'top1':int(bool(ordered) and ordered[0] in gold),
     'origin_top1':int(bool(ordered) and ordered[0]==origin),
     'top5':int(bool(gold & set(ordered[:5]))),
     'origin_top5':int(origin in ordered[:5]),
     'first_gold_rank':next((k+1 for k,j in enumerate(ordered) if j in gold),0),
     'top1_page':D[ordered[0]]['id'] if ordered else None,
     'candidate_pool': [D[j]['id'] for j in arr],
     'candidate_count':len(arr)}
 per.append(entry)
 if (i+1)%50==0:print('Ranked',i+1,'of',len(Q),'elapsed',round(time.time()-st,1),flush=True)
with (OUT/'per_query.jsonl').open('w') as f:
 for x in per:f.write(json.dumps(x,ensure_ascii=False)+'\n')

# Pair-by-issue inferential analysis; 2 newspaper strata, equal weighting.
records=collections.defaultdict(dict)
for r in per:records[r['anchor_id']][r['form']]=r
assert len(records)==60 and all(set(v)==set(forms) for v in records.values())
anchors={s:sorted(a for a in records if records[a]['clean']['source']==s) for s in ('IMVO','ISIG')}
assert all(len(z)==30 for z in anchors.values())
B=10000;rng=np.random.default_rng(2026102422)
bsidx={s:rng.integers(0,len(anchors[s]),size=(B,len(anchors[s]))) for s in anchors}

def stat(form,method,metric):
 groupmeans={};boots=[]
 for s in ('IMVO','ISIG'):
  z=np.asarray([records[a][form][method][metric] for a in anchors[s]],dtype=np.float32)
  groupmeans[s]=float(z.mean());boots.append(z[bsidx[s]].mean(axis=1))
 return {'macro':float(np.mean(list(groupmeans.values()))), 'by_source':groupmeans,
         'ci95':[float(v) for v in np.quantile(np.mean(boots,axis=0),[.025,.975])]}

def contrast(form,ma,mb,metric):
 groupmeans={};boots=[]
 for s in ('IMVO','ISIG'):
  z=np.asarray([records[a][form][ma][metric]-records[a][form][mb][metric] for a in anchors[s]],dtype=np.float32)
  groupmeans[s]=float(z.mean());boots.append(z[bsidx[s]].mean(axis=1))
 return {'macro':float(np.mean(list(groupmeans.values()))), 'by_source':groupmeans,
         'ci95':[float(v) for v in np.quantile(np.mean(boots,axis=0),[.025,.975])]}

metrics={}
for form in forms:
 metrics[form]={}
 for method in methods:
  metrics[form][method]={metric:stat(form,method,metric) for metric in ('recall20','top1','top5','origin_top1')}

hyp={
 'H21a_edit2_char20_minus_word20_recall20':contrast('edit2_boundary','char20','word20','recall20'),
 'H21b_edit2_hybrid_fuzzy_minus_word_fuzzy_top1':contrast('edit2_boundary','hybrid15w5c_fuzzy','word20_fuzzy','top1'),
 'H21c_keyword_hybrid_minus_word_top1':contrast('keyword','hybrid15w5c','word20','top1')
}
hyp['H21a_edit2_char20_minus_word20_recall20']['pass']=bool(hyp['H21a_edit2_char20_minus_word20_recall20']['macro']>=.10 and hyp['H21a_edit2_char20_minus_word20_recall20']['ci95'][0]>0)
hyp['H21b_edit2_hybrid_fuzzy_minus_word_fuzzy_top1']['pass']=bool(hyp['H21b_edit2_hybrid_fuzzy_minus_word_fuzzy_top1']['macro']>=.05 and hyp['H21b_edit2_hybrid_fuzzy_minus_word_fuzzy_top1']['ci95'][0]>0 and min(hyp['H21b_edit2_hybrid_fuzzy_minus_word_fuzzy_top1']['by_source'].values())>=-.05)
hyp['H21c_keyword_hybrid_minus_word_top1']['pass']=bool(min(hyp['H21c_keyword_hybrid_minus_word_top1']['by_source'].values())>=-.05)

# Diagnostics separate genuine recalled evidence from rank rearrangements.
recovery={}
for f in forms:
 recovery[f]={}
 for a,b in [('char20','word20'),('hybrid15w5c_fuzzy','word20_fuzzy'),('hybrid15w5c','word20')]:
  key=a+'_vs_'+b
  recovery[f][key]={'gain':sum(r[a]['top1']>r[b]['top1'] for r in per if r['form']==f),
                    'loss':sum(r[a]['top1']<r[b]['top1'] for r in per if r['form']==f),
                    'newly_recalled':sum(r[a]['recall20']>r[b]['recall20'] for r in per if r['form']==f),
                    'recall_displaced':sum(r[a]['recall20']<r[b]['recall20'] for r in per if r['form']==f)}

# Fixed within-source shuffled gold: a single permutation of all 60 anchors.
null={};rng_null=np.random.default_rng(2026102423);replacement={}
for s in anchors:
 pp=rng_null.permutation(len(anchors[s]))
 for i,a in enumerate(anchors[s]):
  replacement[a]=set(records[anchors[s][int(pp[i])]]['clean']['gold_pages'])
for f in forms:
 null[f]={}
 for method in methods:
  null[f][method]=sum(records[a][f][method]['top1_page'] in replacement[a] for a in records)/60.

result={'protocol_sha256':hashlib.sha256((OUT/'PRECOMMIT.md').read_bytes()).hexdigest(),
 'query_sha256':hashlib.sha256((OUT/'queries_frozen.jsonl').read_bytes()).hexdigest(),
 'n_pages':len(D),'n_issues':len(records),'n_queries':len(Q),
 'methods':list(methods),'forms':list(forms),'primary_metrics':metrics,
 'hypotheses':hyp,'recovery_diagnostics':recovery,'null_shuffled_anchor_top1':null,
 'independent_human_relevance':False,'authentic_ocr':False,'known_exact_paragraph_evidence_gold':True,
 'method_novelty_claim':False}
(OUT/'phase21_results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
print('--- HYPOTHESES ---',json.dumps(hyp,indent=2),flush=True)
for form in forms:
 print('FORM',form,flush=True)
 for method in methods:
  r=metrics[form][method]
  print(method,'recall20',round(100*r['recall20']['macro'],1),'top1',round(100*r['top1']['macro'],1),flush=True)
print('Phase21 complete',flush=True)
