"""Frozen Phase24 retrieval test. Never modifies query/anchor selection.
All results derive from prospective sample / exact evidence gold constructed before scoring.
"""
from __future__ import annotations
import collections,csv,hashlib,json,re,time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from rapidfuzz.fuzz import partial_ratio
from sklearn.feature_extraction.text import CountVectorizer
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'phase24';T=time.time()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for fn,shafile in [('PRECOMMIT.md','PRECOMMIT_SHA256.txt'),('sample_frozen.jsonl','SAMPLE_SHA256.txt'),('queries_gold_frozen.jsonl','GOLD_SHA256.txt')]:
 assert sha(OUT/fn)==(OUT/shafile).read_text().split()[0],f'FREEZE FAILURE: {fn}'
D=[json.loads(x) for x in (ROOT/'pages.jsonl').open()]
Q=[json.loads(x) for x in (OUT/'queries_gold_frozen.jsonl').open()]
assert len(D)==3608 and len(Q)==1436
pidx={p['id']:j for j,p in enumerate(D)}
qtext=[q['query'] for q in Q]
# Exact Phase23 BM25 formula.
cv=CountVectorizer(lowercase=True,token_pattern=r'(?u)\b[^\W\d_]+\b',dtype=np.float32)
X=cv.fit_transform([d['raw'] for d in D]).tocsr();n=len(D)
lens=np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
df=np.asarray((X>0).sum(axis=0)).ravel()
idf=np.log1p((n-df+.5)/(df+.5))
avg=max(float(lens.mean()),1.)
C=X.tocoo()
w=C.data*2.2/(C.data+1.2*(.25+.75*lens[C.row]/avg))*idf[C.col]
matrix=sp.csr_matrix((w,(C.row,C.col)),shape=C.shape,dtype=np.float32)
qmat=cv.transform(qtext);qmat.data[:]=1
scores=(qmat@matrix.T).toarray().astype(np.float32)
rank=np.argsort(-scores,axis=1,kind='stable')
print('BM25 scored',len(Q),'x',len(D),'in',round(time.time()-T,1),'seconds',flush=True)
clean=lambda s:re.sub('[^a-z0-9]','',s.lower())
raw_norm=[clean(p['raw']) for p in D]
qn=[clean(s) for s in qtext]
rows=[]
for i,q in enumerate(Q):
 gold={pidx[k] for k in q['exact_gold_pages']}
 origin=pidx[q['page_id']]
 pool=[int(j) for j in rank[i] if scores[i,j]>0][:20]
 fuzzy={j:float(partial_ratio(qn[i],raw_norm[j])) for j in pool}
 rerank=sorted(pool,key=lambda j:(-fuzzy[j],-float(scores[i,j]),j))
 m={}
 for met,ordered in [('BM25',pool),('BM25_FUZZY20',rerank)]:
  m[met]={
   'origin_top1':int(bool(ordered) and ordered[0]==origin),
   'evidence_top1':int(bool(ordered) and ordered[0] in gold),
   'origin_recall20':int(origin in pool),
   'evidence_recall20':int(any(j in gold for j in pool)),
   'top1_page':D[ordered[0]]['id'] if ordered else None,
   'top1_score':float(scores[i,ordered[0]]) if ordered else None,
   'top1_fuzzy_score':fuzzy[ordered[0]] if met=='BM25_FUZZY20' and ordered else None,
   'ranked_top20':[D[j]['id'] for j in ordered],
   'candidate_pool':[D[j]['id'] for j in pool],
  }
  assert m[met]['origin_top1']<=m[met]['evidence_top1']
 r={**q,'methods':m}
 rows.append(r)
 if (i+1)%300==0:
  print('ranked',i+1,'seconds',round(time.time()-T,1),flush=True)
with (OUT/'per_query.jsonl').open('w') as f:
 for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
np.savez_compressed(OUT/'phase24_bm25_scores.npz',raw_bm25=scores)
# Source-stratified paired issue bootstrap, two frame outcomes always kept in same issue clusters.
SOURCES=('IMVO','ISIG');FRAMES=('A','B');METHODS=('BM25','BM25_FUZZY20')
issues={s:sorted({q['issue'] for q in rows if q['source']==s}) for s in SOURCES}
lookup={(r['issue'],r['frame']):r for r in rows}
assert len(issues['IMVO'])==517 and len(issues['ISIG'])==201
assert all((issue,frame) in lookup for s in SOURCES for issue in issues[s] for frame in FRAMES)
B=10000;rng=np.random.default_rng(2026102424)
bs={s:rng.integers(0,len(issues[s]),size=(B,len(issues[s]))) for s in SOURCES}
def avg(fn,frame):
 by={};boot=[]
 for s in SOURCES:
  vals=np.asarray([fn(lookup[(issue,frame)]) for issue in issues[s]],float)
  by[s]=float(vals.mean());boot.append(vals[bs[s]].mean(axis=1))
 m=.5*(by['IMVO']+by['ISIG'])
 lo,hi=np.quantile(.5*(boot[0]+boot[1]),[.025,.975])
 return {'macro':float(m),'by_source':by,'ci95':[float(lo),float(hi)]}
metrics={};flat=[]
for frame in FRAMES:
 metrics[frame]={}
 stats={
 'cross_issue_repeat':avg(lambda r:r['cross_issue_repeat'],frame),
 'any_other_page':avg(lambda r:r['exact_gold_page_count']>=2,frame),
 'same_issue_multipage_only':avg(lambda r:r['within_issue_multipage_repeat'],frame),
 'bm25_origin_top1':avg(lambda r:r['methods']['BM25']['origin_top1'],frame),
 'bm25_evidence_top1':avg(lambda r:r['methods']['BM25']['evidence_top1'],frame),
 'fuzzy_origin_top1':avg(lambda r:r['methods']['BM25_FUZZY20']['origin_top1'],frame),
 'fuzzy_evidence_top1':avg(lambda r:r['methods']['BM25_FUZZY20']['evidence_top1'],frame),
 'bm25_origin_recall20':avg(lambda r:r['methods']['BM25']['origin_recall20'],frame),
 'bm25_evidence_recall20':avg(lambda r:r['methods']['BM25']['evidence_recall20'],frame),
 'fuzzy_origin_recall20':avg(lambda r:r['methods']['BM25_FUZZY20']['origin_recall20'],frame),
 'fuzzy_evidence_recall20':avg(lambda r:r['methods']['BM25_FUZZY20']['evidence_recall20'],frame),
 'bm25_evidence_minus_origin':avg(lambda r:r['methods']['BM25']['evidence_top1']-r['methods']['BM25']['origin_top1'],frame),
 'fuzzy_evidence_minus_origin':avg(lambda r:r['methods']['BM25_FUZZY20']['evidence_top1']-r['methods']['BM25_FUZZY20']['origin_top1'],frame),
 'fuzzy_minus_bm25_origin':avg(lambda r:r['methods']['BM25_FUZZY20']['origin_top1']-r['methods']['BM25']['origin_top1'],frame),
 'fuzzy_minus_bm25_evidence':avg(lambda r:r['methods']['BM25_FUZZY20']['evidence_top1']-r['methods']['BM25']['evidence_top1'],frame),
 'model_comparison_diff_in_diff':avg(lambda r:(r['methods']['BM25_FUZZY20']['evidence_top1']-r['methods']['BM25']['evidence_top1'])-(r['methods']['BM25_FUZZY20']['origin_top1']-r['methods']['BM25']['origin_top1']),frame),
 }
 metrics[frame]=stats
 for metric,val in stats.items():
  flat.append({'frame':frame,'metric':metric,'macro':val['macro'],'IMVO':val['by_source']['IMVO'],
     'ISIG':val['by_source']['ISIG'],'ci_lower':val['ci95'][0],'ci_upper':val['ci95'][1]})
with (OUT/'phase24_summary.csv').open('w',newline='') as f:
 dw=csv.DictWriter(f,fieldnames=list(flat[0]));dw.writeheader();dw.writerows(flat)
def result(status,pass_all,each):return {'classification':status,'passes_all_gates':pass_all,'frames':each}
h24a={fr: {'rate':metrics[fr]['cross_issue_repeat'],'point_at_least_5pct':metrics[fr]['cross_issue_repeat']['macro']>=.05,
 'lower95_above_2pct':metrics[fr]['cross_issue_repeat']['ci95'][0]>.02} for fr in FRAMES}
a_pass=all(x['point_at_least_5pct'] and x['lower95_above_2pct'] for x in h24a.values())
h24b={fr:{'gap':metrics[fr]['bm25_evidence_minus_origin'],'point_at_least_5pp':metrics[fr]['bm25_evidence_minus_origin']['macro']>=.05,
 'lower95_above_zero':metrics[fr]['bm25_evidence_minus_origin']['ci95'][0]>0} for fr in FRAMES}
b_pass=all(x['point_at_least_5pp'] and x['lower95_above_zero'] for x in h24b.values())
h24c={fr:{'did':metrics[fr]['model_comparison_diff_in_diff'],
 'absolute_at_least_5pp':abs(metrics[fr]['model_comparison_diff_in_diff']['macro'])>=.05,
 'CI_excludes_zero':(metrics[fr]['model_comparison_diff_in_diff']['ci95'][0]>0 or metrics[fr]['model_comparison_diff_in_diff']['ci95'][1]<0)} for fr in FRAMES}
c_pass=all(x['absolute_at_least_5pp'] and x['CI_excludes_zero'] for x in h24c.values())
checks={
 'H24a':result('SUPPORTED' if a_pass else 'NOT SUPPORTED',a_pass,h24a),
 'H24b':result('SUPPORTED' if b_pass else 'NOT SUPPORTED',b_pass,h24b),
 'H24c':result('SUPPORTED' if c_pass else 'NOT SUPPORTED',c_pass,h24c),
}
# Sensitivity from 31 untouched Imvo issues, defined before scoring and NOT used for hypothesis gates.
source_summary=json.loads((OUT/'construction_summary.json').read_text())
never=source_summary['previously_unused_issues_by_source']['IMVO']
assert len(never)==31 and not source_summary['previously_unused_issues_by_source']['ISIG']
sensitivity={}
for fr in FRAMES:
 R=[lookup[(issue,fr)] for issue in never]
 sensitivity[fr]={'n_issues':len(R),'repeat_rate':float(np.mean([r['cross_issue_repeat'] for r in R])),
 'bm25_source_top1':float(np.mean([r['methods']['BM25']['origin_top1'] for r in R])),
 'bm25_evidence_top1':float(np.mean([r['methods']['BM25']['evidence_top1'] for r in R])),
 'fuzzy_source_top1':float(np.mean([r['methods']['BM25_FUZZY20']['origin_top1'] for r in R])),
 'fuzzy_evidence_top1':float(np.mean([r['methods']['BM25_FUZZY20']['evidence_top1'] for r in R]))}
result_all={'phase':24,'precommit_hash':sha(OUT/'PRECOMMIT.md'),'sample_hash':sha(OUT/'sample_frozen.jsonl'),
 'gold_hash':sha(OUT/'queries_gold_frozen.jsonl'),'n_corpus_pages':3608,'n_issues':718,'n_queries':len(rows),
 'source_issues':{s:len(issues[s]) for s in SOURCES},
 'sampling_frames':{'A':'uniform_eligible_page_then_paragraph_then_anchor','B':'uniform_eligible_anchor_position_within_issue'},
 'metrics':metrics,'hypotheses':checks,'untouched_issues_imvo':sensitivity,
 'human_relevance_labels':False,'independent_queries':False,'raw_ocr_benchmark':False,
 'summary_note':'Precommitted sampling was NOT conditioned on exact duplicate presence; no new retrieval algorithm was tuned.'}
(OUT/'phase24_results.json').write_text(json.dumps(result_all,ensure_ascii=False,indent=2))
print('PRECOMMITTED TESTS',json.dumps(checks,indent=2),flush=True)
print('UNTOUCHED IMVO',json.dumps(sensitivity,indent=2),flush=True)
print('PHASE24 COMPLETE',round(time.time()-T,1),'sec',flush=True)
