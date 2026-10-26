"""Phase 20 sealed-new-issues evaluation. Do not tune methods after running."""
from __future__ import annotations
import collections, hashlib, json, math, random, re, time
from pathlib import Path
import numpy as np
import regex as rx
from rapidfuzz.fuzz import partial_ratio
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
R=Path(__file__).resolve().parents[1];O=R/'phase20'
D=[json.loads(x) for x in (R/'pages.jsonl').open()]
Q=[json.loads(x) for x in (O/'queries_frozen.jsonl').open()]
P18=[json.loads(x) for x in (R/'isiXhosa_phase18_evidence_routing/balanced_query_results.jsonl').open()]
P19=[json.loads(x) for x in (R/'phase19/queries_precommit.jsonl').open()]
assert len(D)==3608 and len(Q)==462 and len({q['issue'] for q in Q})==154
assert len({q['anchor_id'] for q in Q})==154
assert set(x['issue'] for x in Q).isdisjoint({x['issue'] for x in P18+P19})
expected_sha=json.loads((O/'construction_summary.json').read_text())['sha256_queries']
assert hashlib.sha256((O/'queries_frozen.jsonl').read_bytes()).hexdigest()==expected_sha
W=rx.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
def words(s):return [w.lower() for w in W.findall(s)]
index={d['id']:i for i,d in enumerate(D)}
assert len(index)==len(D)
views=('en','xh');groups=('IMVO_en','ISIG_en','IMVO_xh','ISIG_xh');forms=('keyword','clean','synthetic_noise')
method_names=('raw_bm25','raw_fullfuzzy','gated_fullfuzzy','unconditional_fullfuzzy','hard_fullfuzzy','raw_local120','gated_local120')
print('PHASE20 | 0 query/document disjointness and frozen hashes PASS',flush=True)

# 13-token gold derived from raw *paragraphs*, never from model ranks.
anchor_by_id={q['anchor_id']:q for q in Q}
first=collections.defaultdict(dict)
for a,q in anchor_by_id.items():
 span=tuple(q['anchor_evidence_tokens'].split());assert len(span)==13
 first[span[0]].setdefault(span,[]).append(a)
G={a:set() for a in anchor_by_id}
for j,d in enumerate(D):
 for p in re.split(r'\n\s*\n',d['raw']):
  toks=words(p)
  for i,t in enumerate(toks):
   if t in first and i+13<=len(toks):
    found=first[t].get(tuple(toks[i:i+13]),[])
    for a in found:G[a].add(j)
assert all(index[z['source_page']] in G[z['anchor_id']] for z in Q)
gcount=collections.Counter(len(z) for z in G.values())
print('PHASE20 | 1 exact within-paragraph gold PASS',dict(gcount),flush=True)

# Silver-view classifier with training issues disjoint from ANY previous/recent evaluation issues.
excluded={q['issue'] for q in P18+P19+Q};training=[];labels=[]
for d in D:
 if d['source']!='IMVO' or not(1884<=d['year']<=1887) or d['issue'] in excluded:continue
 for lang in ('en','xh'):
  seq=words(d[lang]);label=int(lang=='xh')
  if len(seq)<10:continue
  rng=random.Random(int(hashlib.sha256((d['id']+lang+'shortcontrol').encode()).hexdigest()[:12],16))
  for i in range(2):
   start=rng.randrange(len(seq)-4);training.append(' '.join(seq[start:start+5]));labels.append(label)
assert len(set(labels))==2
v=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,max_features=80000,sublinear_tf=True,dtype=np.float32)
clf=LogisticRegression(C=2.,class_weight='balanced',solver='liblinear',max_iter=300,random_state=20261023)
clf.fit(v.fit_transform(training),labels)
predict_xh=clf.predict(v.transform([q['query'] for q in Q])).astype(bool)
print('PHASE20 | 2 LID fitted without evaluation issues',len(training),'5-token samples',flush=True)
del clf,v,training,labels

# Fullpage BM25 identical implementation as Phase18/19; no new feature or per-condition tuning.
n=len(D);qtext=[q['query'] for q in Q]
scores={}
for field in ('raw','en','xh'):
 t0=time.time()
 cv=CountVectorizer(lowercase=True,token_pattern=r'(?u)\b[^\W\d_]+\b',dtype=np.float32)
 X=cv.fit_transform([d[field] for d in D]).tocsr()
 ln=np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
 df=np.asarray((X>0).sum(axis=0)).ravel();idf=np.log1p((n-df+.5)/(df+.5))
 mu=max(float(ln.mean()),1.)
 COO=X.tocoo()
 weights=COO.data*2.2/(COO.data+1.2*(.25+.75*ln[COO.row]/mu))*idf[COO.col]
 Wbm=sp.csr_matrix((weights,(COO.row,COO.col)),shape=COO.shape,dtype=np.float32)
 T=cv.transform(qtext);T.data[:]=1.
 scores[field]=(T@Wbm.T).toarray().astype(np.float32)
 print('PHASE20 | 3 BM25',field,'shape',X.shape,'nnz',X.nnz,'secs',round(time.time()-t0,2),flush=True)
 del cv,X,ln,df,idf,mu,COO,weights,Wbm,T
np.savez_compressed(O/'phase20_bm25_scores.npz',**scores)
rank={field:np.argsort(-sc,axis=1,kind='stable') for field,sc in scores.items()}

def ranked(i,field,count):
 score=scores[field][i]
 return [int(j) for j in rank[field][i] if score[j]>0][:count]

def pool(i,raw_n,en_n,xh_n):
 chosen=[];seen=set()
 for f,nn in [('raw',raw_n),('en',en_n),('xh',xh_n)]:
  if nn:
   for j in ranked(i,f,nn):
    if j not in seen:chosen.append(j);seen.add(j)
 for j in ranked(i,'raw',60):
  if len(chosen)>=20:break
  if j not in seen:chosen.append(j);seen.add(j)
 return chosen[:20]

# Exact dev-selected local measure (Phase19), 120 token window, unique query terms length >=3.
token_cache={}
def local_coverage(q,i):
 qt=set(words(q));qt={t for t in qt if len(t)>=3}
 if not qt:return 0.0
 if i not in token_cache:token_cache[i]=words(D[i]['raw'])
 text=token_cache[i]
 positions=[(k,t) for k,t in enumerate(text) if t in qt]
 if not positions:return 0.0
 counts=collections.defaultdict(int);left=0;best=0
 for pos,t in positions:
  counts[t]+=1
  while positions[left][0]<pos-119:
   old=positions[left][1];counts[old]-=1
   if not counts[old]:del counts[old]
   left+=1
  best=max(best,len(counts))
 return best/len(qt)

norm=re.compile(r'[^a-z0-9]')
page_norm=[norm.sub('',d['raw'].lower()) for d in D]
query_norm=[norm.sub('',q.lower()) for q in qtext]
per=[];start=time.time()
for i,q in enumerate(Q):
 raw20=ranked(i,'raw',20)
 gated=pool(i,20,0,0) if predict_xh[i] else pool(i,15,5,0)
 pools={
  'raw_bm25':raw20,
  'raw_fullfuzzy':raw20,
  'gated_fullfuzzy':gated,
  'unconditional_fullfuzzy':pool(i,15,5,0),
  'hard_fullfuzzy':ranked(i,'xh' if predict_xh[i] else 'en',20),
  'raw_local120':raw20,
  'gated_local120':gated
 }
 fuzz_cache={};local_cache={}
 rec={'qid':q['qid'],'anchor_id':q['anchor_id'],'form':q['form'],'query':q['query'],
   'source':q['source'],'view':q['view'],'group':q['source']+'_'+q['view'],
   'issue':q['issue'],'source_page':q['source_page'],'predict_xh_view':bool(predict_xh[i]),
   'gold_count':len(G[q['anchor_id']]),'gold_pages':[D[j]['id'] for j in sorted(G[q['anchor_id']])],
   'candidate_pools':{name:[D[j]['id'] for j in arr] for name,arr in pools.items()}}
 for name in method_names:
  arr=pools[name]
  if name=='raw_bm25':ordered=arr
  elif name.endswith('fullfuzzy'):
   for j in arr:
    if j not in fuzz_cache:fuzz_cache[j]=partial_ratio(query_norm[i],page_norm[j])
   ordered=sorted(arr,key=lambda j:(-fuzz_cache[j],-float(scores['raw'][i,j]),j))
  else:
   for j in arr:
    if j not in local_cache:local_cache[j]=local_coverage(q['query'],j)
   ordered=sorted(arr,key=lambda j:(-local_cache[j],-float(scores['raw'][i,j]),j))
  g=G[q['anchor_id']];aa=index[q['source_page']]
  rec[name]={'top1':int(bool(ordered) and ordered[0] in g),
             'top5':int(bool(set(ordered[:5])&g)),
             'anchor_top1':int(bool(ordered) and ordered[0]==aa),
             'anchor_top5':int(aa in ordered[:5]),
             'recall20':int(bool(set(arr)&g)),
             'anchor_recall20':int(aa in arr),
             'top1_page':D[ordered[0]]['id'] if ordered else None,
             'anchor_rank':next((k+1 for k,j in enumerate(ordered) if j==aa),0),
             'first_gold_rank':next((k+1 for k,j in enumerate(ordered) if j in g),0),
             'candidate_pool_size':len(arr)}
 per.append(rec)
 if (i+1)%60==0:print('PHASE20 | 4 scored',i+1,'/',len(Q),'queries','elapsed',round(time.time()-start,1),flush=True)
with (O/'per_query_results.jsonl').open('w') as f:
 for r in per:f.write(json.dumps(r,ensure_ascii=False)+'\n')

# Four-cell *issue*-cluster paired bootstrap: every anchor/issue sampled as a unit, preserving query forms.
anchors=sorted(anchor_by_id);r_by_anchor=collections.defaultdict(dict)
for r in per:r_by_anchor[r['anchor_id']][r['form']]=r
for a,d in r_by_anchor.items():assert set(d)==set(forms)
ids_by_group={g:[a for a in anchors if anchor_by_id[a]['source']+'_'+anchor_by_id[a]['view']==g] for g in groups}
assert [len(ids_by_group[g]) for g in groups]==[50,4,50,50]
B=10000;rng=np.random.default_rng(2026102421)
boots={g:rng.integers(0,len(ids_by_group[g]),size=(B,len(ids_by_group[g]))) for g in groups}

def by_group_stat(f):
 points={};samps=[]
 for g in groups:
  vals=np.array([f(r_by_anchor[a]) for a in ids_by_group[g]],dtype=np.float32)
  points[g]=float(vals.mean());samps.append(vals[boots[g]].mean(axis=1))
 return points, np.mean(samps,axis=0)

def summary_stat(f):
 points,bs=by_group_stat(f)
 return {'macro':float(np.mean(list(points.values()))),
         'by_group':points,
         'ci95':[float(z) for z in np.quantile(bs,[.025,.975])],
         'n_issues_by_group':{g:len(ids_by_group[g]) for g in groups},
         'secondary_three_cell_macro':float(np.mean([points[g] for g in groups if g!='ISIG_en'])),
         'secondary_three_cell_ci95':[float(z) for z in np.quantile(np.mean([np.array([f(r_by_anchor[a]) for a in ids_by_group[g]],dtype=np.float32)[boots[g]].mean(axis=1) for g in groups if g!='ISIG_en'],axis=0),[.025,.975])]}

result={'protocol_sha256':hashlib.sha256((O/'PRECOMMIT.md').read_bytes()).hexdigest(),
        'amendment_sha256':hashlib.sha256((O/'FEASIBILITY_AMENDMENT.md').read_bytes()).hexdigest(),
        'query_sha256':expected_sha,
        'setup':{'total_pages':len(D),'n_source_issues':len(anchors),'n_queries':len(Q),
                 'cell_n':{g:len(ids_by_group[g]) for g in groups},'gold_distribution':dict(gcount),
                 'silver_lid_predictions_by_form_view':{f+'_'+v:{'n':len([r for r in per if r['form']==f and r['view']==v]),'xh_fraction':float(np.mean([r['predict_xh_view'] for r in per if r['form']==f and r['view']==v]))} for f in forms for v in views}},
        'method_metrics':{},'comparisons':{},'decision':{},'limitations':[]}
for form in forms:
 for method in method_names:
  result['method_metrics'][form+'_'+method]={met:summary_stat(lambda a,m=met,f=form,x=method:a[f][x][m]) for met in ('anchor_top1','top1','anchor_top5','top5','recall20')}
 for method in method_names:
  if method=='raw_bm25':continue
  result['comparisons'][form+'_'+method+'_minus_raw_bm25']=summary_stat(lambda a,f=form,x=method:a[f][x]['anchor_top1']-a[f]['raw_bm25']['anchor_top1'])

result['comparisons']['H20a_interaction'] = summary_stat(lambda a:(a['synthetic_noise']['raw_fullfuzzy']['anchor_top1']-a['synthetic_noise']['raw_bm25']['anchor_top1']) -(a['keyword']['raw_fullfuzzy']['anchor_top1']-a['keyword']['raw_bm25']['anchor_top1']))
result['comparisons']['H20b_keyword_local_minus_bm25']=summary_stat(lambda a:a['keyword']['raw_local120']['anchor_top1']-a['keyword']['raw_bm25']['anchor_top1'])
H_a=result['comparisons']['H20a_interaction'];H_b=result['comparisons']['H20b_keyword_local_minus_bm25']
result['decision']['H20a_pass']=bool(H_a['macro']>=.1 and H_a['ci95'][0]>0)
result['decision']['H20b_pass']=bool(H_b['macro']>=.05 and H_b['ci95'][0]>0 and min(H_b['by_group'].values())>=-.05)
result['decision']['no_methods_novelty_inference']=True

# Frozen source-view within-stratum anchor shuffle; model predictions never used in shuffling.
null={}
for form in forms:
 for method in method_names:
  vals=[]
  for g in groups:
   ids=ids_by_group[g];rngg=np.random.default_rng(2026102422+groups.index(g))
   shuffled=rngg.permutation([r_by_anchor[a][form]['source_page'] for a in ids])
   vals.append(float(np.mean([r_by_anchor[a][form][method]['top1_page']==shuffled[k] for k,a in enumerate(ids)])))
  null[form+'_'+method]=float(np.mean(vals))
result['null_anchor_top1']=null
result['diagnostics']={}
for form in forms:
 result['diagnostics'][form]={
  'bm25_hit_fuzzy_miss':int(sum(a[form]['raw_bm25']['anchor_top1'] and not a[form]['raw_fullfuzzy']['anchor_top1'] for a in r_by_anchor.values())),
  'bm25_miss_fuzzy_hit':int(sum(not a[form]['raw_bm25']['anchor_top1'] and a[form]['raw_fullfuzzy']['anchor_top1'] for a in r_by_anchor.values())),
  'bm25_hit_local_miss':int(sum(a[form]['raw_bm25']['anchor_top1'] and not a[form]['raw_local120']['anchor_top1'] for a in r_by_anchor.values())),
  'bm25_miss_local_hit':int(sum(not a[form]['raw_bm25']['anchor_top1'] and a[form]['raw_local120']['anchor_top1'] for a in r_by_anchor.values())),
  'gated_pool_changes':int(sum(set(a[form]['candidate_pools']['gated_fullfuzzy'])!=set(a[form]['candidate_pools']['raw_bm25']) for a in r_by_anchor.values()))}
result['limitations']=[
 'Source-derived keyword bags are not independently authored human information needs or semantic paraphrase judgments.',
 'Gold is anchored by a source paragraph and repeated exact 13-token evidence, not exhaustive relevance adjudication.',
 'Synthetic two-character plus word-boundary changes are not measured authentic historical OCR errors.',
 'Archive English and Xhosa views are source labels, not qualified sentence- or word-level language annotations.',
 'Isigidimi English-view independent pool had only four eligible new issue anchors after strict source exclusions.',
 'All methods index the same collection as earlier studies, as required for archive search; index presence is not held-out training label leakage.',
 'No pretrained multilingual dense/hybrid retriever available locally; no genuine external human evaluation or aligned scan benchmark.'
]
(O/'phase20_results.json').write_text(json.dumps(result,indent=2))
print('PHASE20 | 5 EVALUATION FINISHED')
for form in forms:
 print('FORM',form)
 for method in method_names:
  z=result['method_metrics'][form+'_'+method]['anchor_top1']
  print(' ',method,'%.2f'%(100*z['macro']),'[%.2f,%.2f]'%(100*z['ci95'][0],100*z['ci95'][1]),'bygroup',{k:round(100*v,1) for k,v in z['by_group'].items()})
for h in ['H20a_interaction','H20b_keyword_local_minus_bm25']:
 z=result['comparisons'][h];print(h,round(z['macro']*100,2),'ci pp',*[round(v*100,2) for v in z['ci95']])
print('DECISION',result['decision'],'minutes',round((time.time()-start)/60,2),flush=True)
