"""Independent verification of frozen query/source matches and saved retrieval decisions."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
import numpy as np
from rapidfuzz import fuzz
import regex as rx
P=Path(__file__).parent;ROOT=P.parent
S=json.loads((P/'phase22_results.json').read_text())
Q=[json.loads(l) for l in (P/'paired_queries_frozen.jsonl').open()]
F=[json.loads(l) for l in (P/'queries_external_frozen.jsonl').open()]
R=[json.loads(l) for l in (P/'per_query.jsonl').open()]
D=[json.loads(l) for l in (ROOT/'pages.jsonl').open()]
assert len(D)==3608 and len(Q)==len(F)==8 and len(R)==16
assert len({d['id'] for d in D})==len(D)
lookup={d['id']:d for d in D};idx={d['id']:i for i,d in enumerate(D)}
sha=lambda f:hashlib.sha256((P/f).read_bytes()).hexdigest()
for filepath,hashpath in [('PRECOMMIT.md','PRECOMMIT_SHA256.txt'),('queries_external_frozen.jsonl','QUERY_SHA256.txt'),('paired_queries_frozen.jsonl','PAIRS_SHA256.txt')]:
 assert sha(filepath)==(P/hashpath).read_text().split()[0],filepath
assert S['external_query_sha256']==sha('queries_external_frozen.jsonl')
assert S['paired_query_sha256']==sha('paired_queries_frozen.jsonl')
assert S['protocol_sha256']==sha('PRECOMMIT.md')
W=rx.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
def tok(x):return [s.lower() for s in W.findall(x)]
for i,q in enumerate(Q):
 assert q['page_id']==F[i]['page_id']
 assert q['source']==lookup[q['page_id']]['source']
 assert q['issue']==lookup[q['page_id']]['issue']
 assert q['query_external']==F[i]['query_external']
 assert len(tok(q['query_external']))<=24
 page=tok(lookup[q['page_id']]['raw'])
 k=q['clean_window_words'];j=q['clean_start_word']
 assert ' '.join(page[j:j+k])==q['query_clean']
 assert abs(k-len(tok(q['query_external'])))<=3
 assert abs(fuzz.ratio(' '.join(tok(q['query_external'])),' '.join(tok(q['query_clean'])))-q['alignment_ratio_0_100'])<1e-8

score=np.load(P/'phase22_scores.npz')
assert set(score.files)==set(['BM25','CHAR3_TFIDF','XH_BM25_FUZZY20'])
for k in score.files:
 assert score[k].shape==(16,3608) and np.isfinite(score[k]).all()
Dnorm=[re.sub('[^a-z0-9]','',d['raw'].lower()) for d in D]
valid=('BM25','BM25_FUZZY20','CHAR3_TFIDF','CHAR3_FUZZY20','XH_BM25_FUZZY20')
for i,row in enumerate(R):
 q=Q[i//2];mode=('external','clean')[i%2]
 assert row['mode']==mode and row['page_id']==q['page_id']
 assert row['text']==q['query_'+mode]
 for m in valid:
  kind='BM25' if m in ['BM25','BM25_FUZZY20'] else ('CHAR3_TFIDF' if m in ['CHAR3_TFIDF','CHAR3_FUZZY20'] else 'XH_BM25_FUZZY20')
  rawscore=score[kind][i]
  order=np.argsort(-rawscore,kind='stable')
  pool=[j for j in order if rawscore[j]>0][:20]
  assert len(pool)<=20
  if 'FUZZY20' in m:
   nq=re.sub('[^a-z0-9]','',row['text'].lower())
   ranks=sorted(range(len(pool)),key=lambda t:(-fuzz.partial_ratio(nq,Dnorm[pool[t]]),t))
   rank=[pool[t] for t in ranks]
  else:rank=pool
  target=idx[q['page_id']]
  saved=row['methods'][m]
  assert saved['candidates']==[D[j]['id'] for j in pool]
  assert saved['ranked']==[D[j]['id'] for j in rank]
  assert saved['top1']==int(bool(rank) and rank[0]==target)
  assert saved['top5']==int(target in rank[:5])
  assert saved['recall20']==int(target in pool)
for mode in ('external','clean'):
 subset=[r for r in R if r['mode']==mode]
 for m in valid:
  for metric in ('top1','top5','recall20'):
   mean=sum(r['methods'][m][metric] for r in subset)/len(subset)
   assert abs(mean-S['metric_by_mode'][mode][m][metric])<1e-12
for m in valid:
 for metric in ('top1','top5','recall20'):
  diff=np.mean([R[2*i]['methods'][m][metric]-R[2*i+1]['methods'][m][metric] for i in range(8)])
  assert abs(diff-S['paired_external_minus_clean'][m][metric]['external_minus_clean'])<1e-12
print('PASS: 8 provenance-matched pages, SHA-frozen protocol and queries, 16 query records, all candidate pools, 80 rerank decisions, and all summary metrics; no authentic raw OCR inferred')
