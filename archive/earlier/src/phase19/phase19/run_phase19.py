"""Phase 19 frozen-protocol evaluation; no methods/parameters fitted to new-query results.

Rebuild the archive first: python ../build_corpus.py
Run: python run_phase19.py
"""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict,Counter
import hashlib,json,random,re,sys,time,os
import numpy as np
import regex as rx
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from rapidfuzz.fuzz import partial_ratio
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'phase19'
Q=list(map(json.loads,(OUT/'queries_precommit.jsonl').open()))
D=list(map(json.loads,(ROOT/'pages.jsonl').open()))
P18=list(map(json.loads,(ROOT/'isiXhosa_phase18_evidence_routing/balanced_query_results.jsonl').open()))
assert len(D)==3608 and len(Q)==55
idx={d['id']:i for i,d in enumerate(D)}
assert len(idx)==len(D)
W=rx.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
def words(s):return [w.lower() for w in W.findall(s)]
raw_texts=[d['raw'] for d in D]
qtext=[r['text'] for r in Q]
issues_test={x['issue'] for x in Q}
issues_p18={z['issue'] for z in P18 if z['split']=='test'}
issues_p18_all={z['issue'] for z in P18}
assert issues_test.isdisjoint(issues_p18_all)
print('0 | data and held-out issue audit PASS',len(Q),len(issues_test),'new issues',flush=True)

# Independent of retrieval scores, literal raw-paragraph evidence-equivalence multi-gold
need=defaultdict(list)
for i,z in enumerate(Q):need[tuple(z['anchor_evidence_tokens'].split())].append(i)
gold=[set() for z in Q]
for j,d in enumerate(D):
 for par in re.split(r'\n\s*\n',d['raw']):
  toks=words(par)
  for i in range(len(toks)-12):
   for k in need.get(tuple(toks[i:i+13]),[]):gold[k].add(j)
assert all(idx[q['source_page']] in g for q,g in zip(Q,gold)), 'Some fixed anchor spans missing in original raw pages'
print('1 | gold anchor audit PASS', Counter(len(g) for g in gold),flush=True)

# Phase18 frozen short-query silver-language classifier: IMVO 1884-1887 excluding Phase18 test and P19 issues.
training=[];labels=[]
for d in D:
 if d['source']!='IMVO' or not(1884<=d['year']<=1887) or d['issue'] in issues_p18|issues_test:continue
 for lang in ('en','xh'):
  ww=words(d[lang]);label=int(lang=='xh')
  if len(ww)<10:continue
  rng=random.Random(int(hashlib.sha256((d['id']+lang+'shortcontrol').encode()).hexdigest()[:12],16))
  for _ in range(2):
   pos=rng.randrange(len(ww)-4)
   training.append(' '.join(ww[pos:pos+5]));labels.append(label)
v=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,max_features=80000,sublinear_tf=True,dtype=np.float32)
clf=LogisticRegression(C=2.,class_weight='balanced',solver='liblinear',max_iter=300,random_state=20261023)
clf.fit(v.fit_transform(training),labels)
predict_xh=clf.predict(v.transform(qtext)).astype(bool)
print('2 | silver LID trained',len(training),'excerpts', 'predict_english',int((~predict_xh).sum()),flush=True)
training_count=len(training)
del v,clf,training,labels

# Identical Phase18 full-vocabulary BM25. All 3,608 index pages, no age filtering.
n=len(D);m=len(Q)
scores={}
for field in ('raw','en','xh'):
 st=time.time()
 vectorizer=CountVectorizer(lowercase=True,token_pattern=r'(?u)\b[^\W\d_]+\b',dtype=np.float32)
 X=vectorizer.fit_transform([d[field] for d in D]).tocsr()
 lens=np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
 df=np.asarray((X>0).sum(axis=0)).ravel()
 idf=np.log1p((n-df+.5)/(df+.5))
 mean=max(float(lens.mean()),1.)
 C=X.tocoo()
 weights=C.data*2.2/(C.data+1.2*(.25+.75*lens[C.row]/mean))*idf[C.col]
 weighted=sp.csr_matrix((weights,(C.row,C.col)),shape=C.shape,dtype=np.float32)
 T=vectorizer.transform(qtext);T.data[:]=1
 scores[field]=(T@weighted.T).toarray().astype(np.float32)
 print('3 | BM25',field,'shape',X.shape,'nnz',X.nnz,'seconds',round(time.time()-st,1),flush=True)
 del vectorizer,X,lens,df,idf,C,weighted,T

np.savez_compressed(OUT/'phase19_bm25_score_matrices.npz',**scores)
# Established character-token spelling-sensitive baseline. No reranker change.
char_scores=None
try:
 if os.environ.get("SKIP_CHAR") == "1": raise MemoryError("Char-TFIDF exceeded previous execution limit; explicitly omitted before result inspection")
 st=time.time()
 vectorizer=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,max_features=200000,sublinear_tf=True,dtype=np.float32)
 X=vectorizer.fit_transform(raw_texts)
 char_scores=(vectorizer.transform(qtext)@X.T).toarray().astype(np.float32)
 print('4 | character TFIDF shape',X.shape,'nnz',X.nnz,'seconds',round(time.time()-st,1),flush=True)
 del X,vectorizer
except MemoryError:
 print('4 | character TFIDF resource limit, comparison omitted',flush=True)

# Candidate extraction and reranking with Phase18 normalization and tie breaks.
norm=re.compile(r'[^a-z0-9]')
normalized_pages=[norm.sub('',s.lower()) for s in raw_texts]
normalized_queries=[norm.sub('',s.lower()) for s in qtext]
rank={k:np.argsort(-s,axis=1,kind='stable') for k,s in scores.items()}
if char_scores is not None:rank['char']=np.argsort(-char_scores,axis=1,kind='stable')

def rank_pos(i,field,num=20):
 arr=scores[field][i]
 return [int(j) for j in rank[field][i] if arr[j]>0][:num]

def build_pool(i,nraw,nen,nxh):
 chosen=[];seen=set()
 for field,cnt in (('raw',nraw),('en',nen),('xh',nxh)):
  if cnt:
   for j in rank_pos(i,field,cnt):
    if j not in seen:chosen.append(j);seen.add(j)
 for j in rank_pos(i,'raw',60):
  if len(chosen)>=20:break
  if j not in seen:chosen.append(j);seen.add(j)
 return chosen[:20]

def rrf_pool(i):
 rrf=np.zeros(n,dtype=np.float32)
 for fld in ('raw','en'):
  good=rank_pos(i,fld,n)
  for rr,j in enumerate(good):rrf[j]+=1/(60+rr+1)
 arr=np.argsort(-rrf,kind='stable')
 return [int(j) for j in arr if rrf[j]>0][:20]

per_query=[]
method_names=['raw_bm25','raw_fuzzy','hard','gated','unconditional','oracle_english_field','rrf_raw_en']+(['char_word_tfidf'] if char_scores is not None else [])
for i,z in enumerate(Q):
 raw20=rank_pos(i,'raw',20)
 cand={'raw_bm25':raw20,'raw_fuzzy':raw20,
       'hard':rank_pos(i,'xh' if predict_xh[i] else 'en',20),
       'gated':(build_pool(i,20,0,0) if predict_xh[i] else build_pool(i,15,5,0)),
       'unconditional':build_pool(i,15,5,0),
       'oracle_english_field':rank_pos(i,'en',20),
       'rrf_raw_en':rrf_pool(i)}
 if char_scores is not None:
  s=char_scores[i];cand['char_word_tfidf']=[int(j) for j in rank['char'][i] if s[j]>0][:20]
 fuzzy_cache={}
 out={'qid':z['qid'],'source':z['source'],'issue':z['issue'],'query':z['text'],
      'source_page':z['source_page'],'gold_page_count':len(gold[i]),'predict_xh_view':bool(predict_xh[i]),
      'gold_pages':[D[j]['id'] for j in sorted(gold[i])]}
 out['candidate_pools']={key:[D[j]['id'] for j in cand[key]] for key in method_names}
 for key in method_names:
  candidates=cand[key]
  if key=='raw_bm25':chosen=candidates
  else:
   for j in candidates:
    if j not in fuzzy_cache:fuzzy_cache[j]=partial_ratio(normalized_queries[i],normalized_pages[j])
   chosen=sorted(candidates,key=lambda j:(-fuzzy_cache[j],-float(scores['raw'][i,j]),j))
  g=gold[i];a=idx[z['source_page']]
  out[key]={'top1':int(bool(chosen) and chosen[0] in g),'top5':int(any(j in g for j in chosen[:5])),
            'recall20':int(any(j in g for j in candidates)),
            'anchor_top1':int(bool(chosen) and chosen[0]==a),
            'anchor_top5':int(a in chosen[:5]),
            'chosen_top1':D[chosen[0]]['id'] if chosen else None,
            'gold_rank_in_candidates':int(min([r for r,j in enumerate(chosen,1) if j in g],default=0)),
            'pool_size':len(candidates)}
 per_query.append(out)
 if (i+1)%10==0:print('5 | evaluated',i+1,'/',len(Q),'queries',flush=True)

# Primary statistical results: sources have unique issues, bootstrapped separately; independent source weights.
seed=20261024;B=10000;rng=np.random.default_rng(seed)
ids_by_source={src:np.array([i for i,z in enumerate(Q) if z['source']==src],dtype=int) for src in ('IMVO','ISIG')}
indexes={src:rng.integers(0,len(ids_by_source[src]),size=(B,len(ids_by_source[src]))) for src in ids_by_source}
stat={};diff_ci={}
for metric in ('top1','top5','recall20','anchor_top1','anchor_top5'):
 matrix={name:np.array([p[name][metric] for p in per_query],dtype=np.float32) for name in method_names}
 method_stats={}
 for name,data in matrix.items():
  v={src:float(data[ix].mean()) for src,ix in ids_by_source.items()}
  method_stats[name]={'source':v,'macro':float(np.mean(list(v.values()))),'micro':float(data.mean())}
 stat[metric]=method_stats
 for base in ('raw_fuzzy','raw_bm25'):
  for method in method_names:
   if method==base:continue
   d=matrix[method]-matrix[base]
   samples=[]
   by_source={}
   for src,ix in ids_by_source.items():
    samp=d[ix][indexes[src]].mean(axis=1)
    samples.append(samp)
    by_source[src]={'point':float(d[ix].mean()),'ci95':[float(x) for x in np.quantile(samp,[.025,.975])]}
   boots=np.mean(samples,axis=0)
   diff_ci[f'{metric}:{method}-minus-{base}']={'point':float(np.mean([d[ix].mean() for ix in ids_by_source.values()])),
      'ci95':[float(x) for x in np.quantile(boots,[.025,.975])], 'by_source':by_source}

# Frozen source-within-group shuffled-gold null. The shuffled *source page* is the pseudo-expected gold;
# the observed nonzero query hits should greatly exceed target-random collisions.
null={}
for method in method_names:
 vals=[]
 for src,ix in ids_by_source.items():
  rngsrc=np.random.default_rng(seed+(1 if src=='IMVO' else 2))
  anchors=[Q[j]['source_page'] for j in ix]
  shifts=rngsrc.permutation(len(ix))
  vals.append(float(np.mean([per_query[int(i)][method]['chosen_top1']==anchors[int(shifts[k])] for k,i in enumerate(ix)])))
 null[method]=float(np.mean(vals))

# Decision according to frozen criterion: a +3 pp threshold and strictly positive CI
x=diff_ci['top1:gated-minus-raw_fuzzy'];y=diff_ci['top1:gated-minus-raw_bm25']
pass_gate=bool(x['point']>=.03 and y['point']>=.03 and x['ci95'][0]>0 and y['ci95'][0]>0 and all(x['by_source'][src]['point']>=-.02 for src in ('IMVO','ISIG')))

res={'dataset':{'num_pages':len(D),'queries':len(Q),'source_counts':{s:int(len(ix)) for s,ix in ids_by_source.items()},
         'all_source_issues_distinct':True,'all_source_issues_disjoint_phase18':True,
         'min_threegram_overlap':0,'multiple_evidence_pages':int(sum(len(g)>1 for g in gold)),
         'annotation_type':'assistant_composed_source_visible English queries; no independent semantic relevance labels'},
     'frozen_precommit_sha256':hashlib.sha256((OUT/'PRECOMMIT.md').read_bytes()).hexdigest(),
     'frozen_query_sha256':hashlib.sha256((OUT/'queries_precommit.jsonl').read_bytes()).hexdigest(),
     'model':{'training_excerpts':training_count,'predict_xhosa_view_count':int(predict_xh.sum()),'predict_english_view_count':int((~predict_xh).sum()),
              'labels':'archive language fields; not expert language IDs'},
     'metrics':stat,'paired_issue_bootstrap':diff_ci,'shuffle_gold_source_macro_top1':null,
     'gate':{'H19a_pass':pass_gate,'requirements':'>=+3pp gated vs both raw_fuzzy and raw_bm25 with issue CI>0, source penalties <=2pp'},
     'limitations':['Original anchor plus repeated exact evidence-spans, not exhaustive semantic relevance adjudication',
                    'Model-composed English queries based on visible source paragraphs, not independent human needs',
                    'Only 15 Isigidimi issue-level queries, mostly factual notices/reviews',
                    'No independently authored isiXhosa queries or real aligned OCR scans',
                    'No multilingual pretrained dense/hybrid baseline available in this portable environment',
                    'Phase18 query policy fixed on older benchmark, but the archive itself is same historical collection']}
(OUT/'phase19_results.json').write_text(json.dumps(res,indent=2))
(OUT/'phase19_per_query.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in per_query))
print('FINISHED',json.dumps({'macro_top1':{k:round(v['macro'],4) for k,v in stat['top1'].items()},
 'gated_vs_rawfuzzy':x,'gated_vs_rawbm25':y,'H19a_pass':pass_gate},indent=2),flush=True)
