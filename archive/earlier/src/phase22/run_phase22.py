"""Execute frozen Phase22 paired external-transcript retrieval pilot. No tuning."""
from __future__ import annotations
import collections,hashlib,json,re,time
from pathlib import Path
import numpy as np
from rapidfuzz import fuzz,distance
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import scipy.sparse as sp

P=Path(__file__).parent;ROOT=P.parent
D=[json.loads(l) for l in (ROOT/'pages.jsonl').open()]
Q=[json.loads(l) for l in (P/'paired_queries_frozen.jsonl').open()]
assert len(D)==3608 and len(Q)==8
sha=lambda name:hashlib.sha256((P/name).read_bytes()).hexdigest()
assert sha('PRECOMMIT.md')==(P/'PRECOMMIT_SHA256.txt').read_text().split()[0]
assert sha('queries_external_frozen.jsonl')==(P/'QUERY_SHA256.txt').read_text().split()[0]
assert sha('paired_queries_frozen.jsonl')==(P/'PAIRS_SHA256.txt').read_text().split()[0]
page2idx={d['id']:i for i,d in enumerate(D)}
queries=[]
for p in Q:
 for mode in ('external','clean'):
  queries.append({'page_id':p['page_id'],'source':p['source'],'mode':mode,'text':p['query_'+mode],
   'prior_origin':p['prior_query_origin_phases'],'alignment':p['alignment_ratio_0_100']})

n=len(D);st=time.time()

def bm25(field):
 cv=CountVectorizer(lowercase=True,token_pattern=r'(?u)\b[^\W\d_]+\b',dtype=np.float32)
 X=cv.fit_transform([d[field] for d in D]).tocsr()
 lens=np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
 df=np.asarray((X>0).sum(axis=0)).ravel()
 idf=np.log1p((n-df+.5)/(df+.5))
 avg=max(float(lens.mean()),1.0)
 c=X.tocoo()
 weight=(c.data*2.2/(c.data+1.2*(.25+.75*lens[c.row]/avg))*idf[c.col]).astype(np.float32)
 W=sp.csr_matrix((weight,(c.row,c.col)),shape=X.shape,dtype=np.float32)
 q=cv.transform([q['text'] for q in queries]);q.data[:]=1.
 scores=(q@W.T).toarray().astype(np.float32)
 print('BM25',field,'vocab',X.shape[1],'nonzeros',X.nnz,'elapsed',round(time.time()-st,2),flush=True)
 return scores,X,cv
word,counts,vectorizer=bm25('raw')
xh,_,_=bm25('xh')

# Mirror Phase21 padded within-word character trigram TFIDF exactly.
vocab=vectorizer.get_feature_names_out();tri_index={};v=[];c=[];ptr=[0]
for w in vocab:
 s='^'+w+'$';counter=collections.Counter(s[i:i+3] for i in range(len(s)-2))
 for g,k in counter.items():
  j=tri_index.setdefault(g,len(tri_index));c.append(j);v.append(k)
 ptr.append(len(c))
M=sp.csr_matrix((np.asarray(v,dtype=np.float32),np.asarray(c,dtype=np.int32),np.asarray(ptr,dtype=np.int32)),shape=(len(vocab),len(tri_index)),dtype=np.float32)
Xgram=counts@M
tr=TfidfTransformer(norm='l2',sublinear_tf=True,smooth_idf=True)
Xgram=tr.fit_transform(Xgram)
reg=re.compile(r'(?u)\b[^\W\d_]+\b')
rows=[];cols=[];values=[]
for i,q in enumerate(queries):
 grams=collections.Counter()
 for w in reg.findall(q['text'].lower()):
  s='^'+w+'$';grams.update(s[t:t+3] for t in range(len(s)-2))
 for g,num in grams.items():
  if g in tri_index:
   rows.append(i);cols.append(tri_index[g]);values.append(num)
qraw=sp.csr_matrix((np.asarray(values,dtype=np.float32),(rows,cols)),shape=(len(queries),len(tri_index)),dtype=np.float32)
char=(tr.transform(qraw)@Xgram.T).toarray().astype(np.float32)
print('char3',Xgram.shape,'index nonzeros',Xgram.nnz,'elapsed',round(time.time()-st,2),flush=True)
del Xgram,M,counts,vectorizer,v,c,ptr,vocab,tr,qraw

scores={'BM25':word,'CHAR3_TFIDF':char,'XH_BM25_FUZZY20':xh}
np.savez_compressed(P/'phase22_scores.npz',**scores)
normalizer=re.compile('[^a-z0-9]')
all_norm=[normalizer.sub('',d['raw'].lower()) for d in D]
methods=('BM25','BM25_FUZZY20','CHAR3_TFIDF','CHAR3_FUZZY20','XH_BM25_FUZZY20')


def first20(sc,i):
 ranked=np.argsort(-sc[i],kind='stable')
 return [int(j) for j in ranked if sc[i,j]>0][:20]

def score_one(i,name):
 if name=='BM25':
  pool=first20(word,i);ranked=pool
 elif name=='CHAR3_TFIDF':
  pool=first20(char,i);ranked=pool
 else:
  sc=xh if name=='XH_BM25_FUZZY20' else (char if name=='CHAR3_FUZZY20' else word)
  pool=first20(sc,i)
  normq=normalizer.sub('',queries[i]['text'].lower())
  rankscores=[fuzz.partial_ratio(normq,all_norm[j]) if normq else 0.0 for j in pool]
  ranked=[pool[t] for t in sorted(range(len(pool)),key=lambda t:(-rankscores[t],t))]
 target=page2idx[queries[i]['page_id']]
 return {'top1':int(len(ranked)>0 and ranked[0]==target),
         'top5':int(target in ranked[:5]),'recall20':int(target in pool),
         'top1_page':D[ranked[0]]['id'] if ranked else None,
         'target_page_rank_in_pool':pool.index(target)+1 if target in pool else None,
         'candidates':[D[j]['id'] for j in pool],
         'ranked':[D[j]['id'] for j in ranked]}

results=[]
for i,q in enumerate(queries):
 res={**q,'methods':{m:score_one(i,m) for m in methods}}
 results.append(res)
with (P/'per_query.jsonl').open('w') as f:
 for r in results:f.write(json.dumps(r,ensure_ascii=False)+'\n')

rng=np.random.default_rng(2026102422);boots=rng.integers(0,8,size=(10000,8))

def calc_metrics(sub):
 return {k:sum(r['methods'][k][metric] for r in sub)/len(sub) for k in methods for metric in []}
summary={}
for mode in ('external','clean'):
 arr=[r for r in results if r['mode']==mode]
 summary[mode]={m:{metric:sum(r['methods'][m][metric] for r in arr)/len(arr) for metric in ('top1','top5','recall20')} for m in methods}
 summary[mode]['n']=len(arr)
paired_metrics={}
for m in methods:
 paired_metrics[m]={}
 for metric in ('top1','top5','recall20'):
  clean=np.array([results[2*i+1]['methods'][m][metric] for i in range(8)],dtype=float)
  external=np.array([results[2*i]['methods'][m][metric] for i in range(8)],dtype=float)
  diff=external-clean
  draws=diff[boots].mean(axis=1)
  paired_metrics[m][metric]={'external_minus_clean':float(diff.mean()),'ci95_descriptive':np.quantile(draws,[.025,.975]).tolist(),
    'paired_gain_count':int((diff>0).sum()),'paired_loss_count':int((diff<0).sum()),
    'by_page':{Q[i]['page_id']:float(diff[i]) for i in range(8)}}

# Reprint-sensitive exact token query count in the 3,608-page corpus, diagnostic only.
comp=re.compile(r'[^a-z0-9]')
reprint=[]
for q in Q:
 normalized=comp.sub('',q['query_clean'].lower())
 pages=[d['id'] for d,ntext in zip(D,all_norm) if normalized and normalized in ntext]
 reprint.append({'page_id':q['page_id'],'n_archive_pages_containing_same_normalized_clean_query':len(pages),
  'pages':pages[:25],'query_external_vs_aligned_clean_normalized_levenshtein':distance.Levenshtein.normalized_distance(
    comp.sub('',q['query_external'].lower()),normalized), 'fuzzy_alignment_ratio':q['alignment_ratio_0_100']})

summary_out={'experiment':'phase22_external_transcript_pilot','protocol_sha256':sha('PRECOMMIT.md'),
 'external_query_sha256':sha('queries_external_frozen.jsonl'),'paired_query_sha256':sha('paired_queries_frozen.jsonl'),
 'n_pages':len(D),'n_independent_source_pages':len(Q),'n_query_pairs':len(Q),
 'methods':list(methods),'metric_by_mode':summary,'paired_external_minus_clean':paired_metrics,
 'alignment_and_duplicate_diagnostics':reprint,
 'authentic_uncorrected_ocr_confirmed':False,'independent_human_queries':False,
 'external_transcripts_source_urls_verified_metadata':True,
 'includes_previous_query_origin_issues':sum(bool(r['prior_query_origin_phases']) for r in Q),
 'elapsed_seconds':round(time.time()-st,1)}
(P/'phase22_results.json').write_text(json.dumps(summary_out,indent=2,ensure_ascii=False))
print('RESULTS',json.dumps(summary,indent=2))
for m in methods:
 print('DELTA',m,'top1',paired_metrics[m]['top1']['external_minus_clean'],'recall',paired_metrics[m]['recall20']['external_minus_clean'])
print('total runtime',round(time.time()-st,1),'s',flush=True)
