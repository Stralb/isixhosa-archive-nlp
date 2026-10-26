"""Phase23: locked BM25 retrieval and prespecified exact-evidence reprint-aware evaluation."""
from __future__ import annotations
import collections,hashlib,json,re,time,csv
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from rapidfuzz.fuzz import partial_ratio
from sklearn.feature_extraction.text import CountVectorizer
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'phase23';t0=time.time()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
precommit=sha(OUT/'PRECOMMIT.md');assert precommit==(OUT/'PRECOMMIT_SHA256.txt').read_text().split()[0]
construction=json.loads((OUT/'construction_summary.json').read_text());assert construction['hashes']['PRECOMMIT.md']==precommit
for fn in ('queries_frozen.jsonl','anchor_audit_frozen.jsonl'):
 assert sha(OUT/fn)==construction['hashes'][fn]
D=[json.loads(x) for x in (ROOT/'pages.jsonl').open()]
Q=[json.loads(x) for x in (OUT/'queries_frozen.jsonl').open()]
assert len(D)==3608 and len(Q)==140
page_index={p['id']:j for j,p in enumerate(D)}
SOURCES=('IMVO','ISIG');FORMS=('literal7','keyword6');TYPES=('repeated','unique')
methods=('BM25','BM25_FUZZY20','XH_BM25_FUZZY20')

def bm25(field):
 n=len(D);cv=CountVectorizer(lowercase=True,token_pattern=r'(?u)\b[^\W\d_]+\b',dtype=np.float32)
 X=cv.fit_transform([d[field] for d in D]).tocsr();lengths=np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
 df=np.asarray((X>0).sum(axis=0)).ravel();idf=np.log1p((n-df+.5)/(df+.5))
 avg=max(float(lengths.mean()),1.)
 C=X.tocoo(); w=C.data*2.2/(C.data+1.2*(.25+.75*lengths[C.row]/avg))*idf[C.col]
 matrix=sp.csr_matrix((w,(C.row,C.col)),shape=C.shape,dtype=np.float32)
 qmat=cv.transform([q['query'] for q in Q]);qmat.data[:]=1
 scores=(qmat@matrix.T).toarray().astype(np.float32)
 print('indexed',field,'vocabulary',X.shape[1], 'nnz',X.nnz,'seconds',round(time.time()-t0,1),flush=True)
 return scores
raw_scores=bm25('raw');xh_scores=bm25('xh')
np.savez_compressed(OUT/'phase23_scores.npz',raw_bm25=raw_scores,xh_bm25=xh_scores)
raw_rank=np.argsort(-raw_scores,axis=1,kind='stable');xh_rank=np.argsort(-xh_scores,axis=1,kind='stable')
clean=lambda s:re.sub('[^a-z0-9]','',s.lower())
raw_norm=[clean(d['raw']) for d in D]
query_norm=[clean(q['query']) for q in Q]
rows=[]
for i,q in enumerate(Q):
 gold={page_index[p] for p in q['gold_pages']}
 origin=page_index[q['source_page']]
 pools={
  'BM25':[int(j) for j in raw_rank[i] if raw_scores[i,j]>0][:20],
  'BM25_FUZZY20':[int(j) for j in raw_rank[i] if raw_scores[i,j]>0][:20],
  'XH_BM25_FUZZY20':[int(j) for j in xh_rank[i] if xh_scores[i,j]>0][:20],
 }
 evals={}
 fuzz_cache={}
 for method,pool in pools.items():
  ordered=pool
  if 'FUZZY' in method:
   for j in pool:
    if j not in fuzz_cache:fuzz_cache[j]=partial_ratio(query_norm[i],raw_norm[j])
   ordered=sorted(pool,key=lambda j:(-fuzz_cache[j],-float(raw_scores[i,j]),j))
  def hit(at,target):return int(bool(set(ordered[:at]) & target))
  def orig(at):return int(origin in ordered[:at])
  evals[method]={'origin_top1':orig(1),'origin_top5':orig(5),'origin_recall20':int(origin in pool),
   'equiv_top1':hit(1,gold),'equiv_top5':hit(5,gold),'equiv_recall20':int(bool(set(pool)&gold)),
   'top1_page':D[ordered[0]]['id'] if ordered else None,
   'ordered_pages':[D[j]['id'] for j in ordered],
   'candidate_pool':[D[j]['id'] for j in pool],
   'top1_fuzzy':float(fuzz_cache[ordered[0]]) if 'FUZZY' in method and ordered else None,
   'first_equiv_rank':next((k+1 for k,j in enumerate(ordered) if j in gold),None),
   'first_origin_rank':next((k+1 for k,j in enumerate(ordered) if j==origin),None)}
 rows.append({'qid':q['qid'],'anchor_id':q['anchor_id'],'form':q['form'],'anchor_type':q['anchor_type'],
  'source':q['source'],'issue':q['issue'],'query':q['query'],
  'source_page':q['source_page'],'gold_pages':q['gold_pages'],
  'gold_count':len(gold),'methods':evals})
 if (i+1)%35==0:print('scored',i+1,'queries in',round(time.time()-t0,1),'sec',flush=True)
with (OUT/'per_query.jsonl').open('w') as f:
 for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')

issues={s:sorted(set(r['issue'] for r in rows if r['source']==s)) for s in SOURCES}
assert len(issues['IMVO'])==28 and len(issues['ISIG'])==7
B=10000;rng=np.random.default_rng(2026102423)
bs={s:rng.integers(0,len(issues[s]),size=(B,len(issues[s]))) for s in SOURCES}
lookup={(r['issue'],r['anchor_type'],r['form']):r for r in rows}
assert all((i,t,f) in lookup for s in SOURCES for i in issues[s] for t in TYPES for f in FORMS)

def avg(metric,method,typ,form):
 res={};boot=[]
 for s in SOURCES:
  z=np.asarray([lookup[(issue,typ,form)]['methods'][method][metric] for issue in issues[s]],dtype=float)
  res[s]=float(z.mean());boot.append(z[bs[s]].mean(axis=1))
 bs_mean=.5*(boot[0]+boot[1])
 return {'macro':.5*(res['IMVO']+res['ISIG']),'by_source':res,'ci95':[float(v) for v in np.quantile(bs_mean,[.025,.975])]}

def contrast(first,second,typ,form,metric):
 res={};boot=[]
 for s in SOURCES:
  z=np.asarray([first(lookup[(issue,typ,form)])-second(lookup[(issue,typ,form)]) for issue in issues[s]],dtype=float)
  res[s]=float(z.mean());boot.append(z[bs[s]].mean(axis=1))
 bsmean=.5*(boot[0]+boot[1])
 return {'macro':.5*(res['IMVO']+res['ISIG']),'by_source':res,'ci95':[float(v) for v in np.quantile(bsmean,[.025,.975])]}

metrics={}
flat=[]
for typ in TYPES:
 metrics[typ]={}
 for form in FORMS:
  metrics[typ][form]={}
  for method in methods:
   metrics[typ][form][method]={m:avg(m,method,typ,form) for m in ('origin_top1','equiv_top1','origin_top5','equiv_top5','origin_recall20','equiv_recall20')}
   for m,v in metrics[typ][form][method].items():
    flat.append({'anchor_type':typ,'form':form,'method':method,'metric':m,'macro':v['macro'],'imvo':v['by_source']['IMVO'],
     'isig':v['by_source']['ISIG'],'ci_lo':v['ci95'][0],'ci_hi':v['ci95'][1]})
with (OUT/'phase23_summary.csv').open('w',newline='') as f:
 wr=csv.DictWriter(f,fieldnames=list(flat[0]));wr.writeheader();wr.writerows(flat)

# H23a pooled miss fraction on repeated keyword6 issues (explicitly enriched sampling).
rep=[lookup[(issue,'repeated','keyword6')] for s in SOURCES for issue in issues[s]]
misses=[r for r in rep if not r['methods']['BM25']['origin_top1']]
recovered=[r for r in misses if r['methods']['BM25']['equiv_top1']]
miss_rate=float(len(recovered)/len(misses)) if misses else None
frac_boot=[]
for b in range(B):
 samp=[]
 for s in SOURCES:
  samp += [lookup[(issues[s][int(j)],'repeated','keyword6')] for j in bs[s][b]]
 ms=[r for r in samp if not r['methods']['BM25']['origin_top1']]
 if ms:
  frac_boot.append(sum(r['methods']['BM25']['equiv_top1'] for r in ms)/len(ms))
h23a={'numerator':len(recovered),'denominator':len(misses),'fraction':miss_rate,
 'ci95':[float(x) for x in np.quantile(frac_boot,[.025,.975])],'pass':len(misses)>=10 and miss_rate>=.50}

# H23b exact vs origin changes BM25-fuzzy-minus-BM25 gap, repeated/keyword, macro issue grouped.
def did(r):
 b=r['methods']['BM25'];f=r['methods']['BM25_FUZZY20']
 return (f['equiv_top1']-b['equiv_top1'])-(f['origin_top1']-b['origin_top1'])
h23b=contrast(did,lambda r:0,'repeated','keyword6','diff-in-diff')
h23b['pass_magnitude']=abs(h23b['macro'])>=.05
h23b['robust_direction']=not(h23b['ci95'][0]<=0<=h23b['ci95'][1])
h23b['strict_gap']=contrast(lambda r:r['methods']['BM25_FUZZY20']['origin_top1'],
                              lambda r:r['methods']['BM25']['origin_top1'],'repeated','keyword6','strict')
h23b['exact_gap']=contrast(lambda r:r['methods']['BM25_FUZZY20']['equiv_top1'],
                              lambda r:r['methods']['BM25']['equiv_top1'],'repeated','keyword6','exact')
# H23c unique anchors exactly one gold page.
unique_control_pass=all(r['methods'][method][f'origin_{metric}']==r['methods'][method][f'equiv_{metric}']
   for r in rows if r['anchor_type']=='unique' for method in methods for metric in ('top1','top5','recall20'))
assert unique_control_pass
h23c={'pass':unique_control_pass,'evaluated_query_method_cases':35*2*len(methods)}

raw_gap=contrast(lambda r:r['methods']['BM25']['equiv_top1'],lambda r:r['methods']['BM25']['origin_top1'],
  'repeated','keyword6','equiv_origin')
# Record exact duplicate issue count, since repeated source pages are NOT semantically annotated.
result={'phase':23,'precommit_sha256':precommit,'query_sha256':sha(OUT/'queries_frozen.jsonl'),
 'anchor_audit_sha256':sha(OUT/'anchor_audit_frozen.jsonl'),
 'n_raw_pages':len(D),'n_issues':35,'n_anchors':70,'n_queries':140,'issue_counts':{s:len(issues[s]) for s in SOURCES},
 'methods':list(methods),'metrics':metrics,'hypotheses':{'H23a':h23a,'H23b':h23b,'H23c':h23c},
 'bm25_equiv_minus_origin_repeated_keyword':raw_gap,
 'gold_labels':'exact_13word_paragraph_contiguous','human_judgments':False,'machine_ocr_verified':False,
 'query_independence':'issue_origin_disjoint_but_source_derived','novel_method_claim':False}
(OUT/'phase23_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print('RESULTS',json.dumps({'H23a':h23a,'H23b':h23b,'H23c':h23c,'raw_gap':raw_gap},indent=2),flush=True)
for form in FORMS:
 for typ in TYPES:
  print('TOP1',typ,form,[(m,round(metrics[typ][form][m]['origin_top1']['macro'],3),round(metrics[typ][form][m]['equiv_top1']['macro'],3)) for m in methods],flush=True)
print('PHASE23 ANALYSIS COMPLETE',round(time.time()-t0,1),'sec',flush=True)
