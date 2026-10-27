"""Independent Phase23: gold, SHA, issue disjointness, BM25 score, rank and metric checks."""
from __future__ import annotations
import collections,hashlib,json,re,time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import CountVectorizer
import regex
ROOT=Path(__file__).resolve().parents[1]; O=ROOT/'phase23';start=time.time()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(O/'PRECOMMIT.md')==(O/'PRECOMMIT_SHA256.txt').read_text().split()[0]
summary=json.loads((O/'construction_summary.json').read_text())
for f in ('queries_frozen.jsonl','anchor_audit_frozen.jsonl'):
 assert sha(O/f)==summary['hashes'][f]
D=[json.loads(x) for x in (ROOT/'pages.jsonl').open()]
Q=[json.loads(x) for x in (O/'queries_frozen.jsonl').open()]
A=[json.loads(x) for x in (O/'anchor_audit_frozen.jsonl').open()]
P=[json.loads(x) for x in (O/'per_query.jsonl').open()]
R=json.loads((O/'phase23_results.json').read_text())
assert len(D)==3608 and len(Q)==len(P)==140 and len(A)==70
assert R['query_sha256']==sha(O/'queries_frozen.jsonl')
assert R['anchor_audit_sha256']==sha(O/'anchor_audit_frozen.jsonl')
assert R['precommit_sha256']==sha(O/'PRECOMMIT.md')
pages={d['id']:d for d in D};index={d['id']:i for i,d in enumerate(D)}
excluded=set()
for f in ('isiXhosa_phase18_evidence_routing/balanced_query_results.jsonl','phase19/queries_precommit.jsonl','phase20/queries_frozen.jsonl','phase21/queries_frozen.jsonl','phase22/queries_external_frozen.jsonl'):
 excluded.update(json.loads(l)['issue'] for l in (ROOT/f).open())
assert not {a['issue'] for a in A}&excluded
issues=collections.defaultdict(set)
for a in A:issues[a['source']].add(a['issue'])
assert {s:len(v) for s,v in issues.items()}=={'IMVO':28,'ISIG':7}
W=regex.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
def words(s):return [x.lower() for x in W.findall(s)]
# Reconstruct gold labels without relying on saved original gold lists or the constructor.
anchors={tuple(a['anchor_tokens']):a['anchor_id'] for a in A}
assert len(anchors)==len(A)
first=collections.defaultdict(dict)
for anchor,aid in anchors.items():first[anchor[0]][anchor]=aid
actual=collections.defaultdict(set)
for d in D:
 for rawpar in re.split(r'\n\s*\n',d['raw']):
  seq=words(rawpar)
  for i in range(len(seq)-12):
   possible=first.get(seq[i])
   if not possible:continue
   aid=possible.get(tuple(seq[i:i+13]))
   if aid is not None: actual[aid].add(d['id'])
for a in A:
 aid=a['anchor_id'];src=pages[a['source_page']]
 assert a['source_page'] in actual[aid]
 assert sorted(actual[aid])==a['gold_pages']
 assert a['issue']==src['issue'] and a['source']==src['source']
 assert a['anchor_type']==('repeated' if len({pages[p]['issue'] for p in actual[aid]})>=2 else 'unique')
 assert (len(actual[aid])==1) if a['anchor_type']=='unique' else True
 para=re.split(r'\n\s*\n',src['xh'])[a['source_para_index']]
 pt=words(para)
 assert tuple(pt[a['anchor_start']:a['anchor_start']+13])==tuple(a['anchor_tokens'])
assert all(q['anchor_id']==p['anchor_id'] and q['qid']==p['qid'] and q['query']==p['query'] for q,p in zip(Q,P))
assert len({q['qid'] for q in Q})==140
print('verified exact paragraph-bounded multigold labels, all 70 anchors',round(time.time()-start,1),flush=True)

# Rebuild raw/Xh BM25 matrices completely and compare numerical scores to the saved scores.
saved=np.load(O/'phase23_scores.npz')
for field,key in [('raw','raw_bm25'),('xh','xh_bm25')]:
 cv=CountVectorizer(lowercase=True,token_pattern=r'(?u)\b[^\W\d_]+\b',dtype=np.float32)
 X=cv.fit_transform(d[field] for d in D).tocsr()
 lens=np.asarray(X.sum(axis=1)).reshape(-1).astype(np.float32)
 freq=np.asarray((X>0).sum(axis=0)).reshape(-1)
 idf=np.log1p((len(D)-freq+.5)/(freq+.5))
 tri=X.tocoo();norm=1.2*(0.25+0.75*lens[tri.row]/max(1,float(lens.mean())))
 wd=tri.data*2.2/(tri.data+norm)*idf[tri.col]
 mat=sp.csr_matrix((wd,(tri.row,tri.col)),shape=X.shape,dtype=np.float32)
 qs=cv.transform(q['query'] for q in Q);qs.data[:]=1
 got=(qs@mat.T).toarray().astype(np.float32)
 err=float(np.max(np.abs(got-saved[key])))
 assert np.allclose(got,saved[key],rtol=3e-5,atol=1e-3),(field,err)
 print('verified independent',field,'BM25',got.shape,'max absolute score difference',err,'sec',round(time.time()-start,1),flush=True)
raw=saved['raw_bm25'];xh=saved['xh_bm25']
raw_sort=np.argsort(-raw,axis=1,kind='stable');xh_sort=np.argsort(-xh,axis=1,kind='stable')
clean=lambda s:re.sub('[^a-z0-9]','',s.lower())
page_text=[clean(d['raw']) for d in D]
normq=[clean(q['query']) for q in Q]
for i,(q,per) in enumerate(zip(Q,P)):
 gold=set(q['gold_pages']);origin=q['source_page']
 for name,matrix,order in [('BM25',raw,raw_sort),('BM25_FUZZY20',raw,raw_sort),('XH_BM25_FUZZY20',xh,xh_sort)]:
  pool=[int(k) for k in order[i] if matrix[i,k]>0][:20]
  sortedpool=pool[:]
  if 'FUZZY' in name:
   sims={j:fuzz.partial_ratio(normq[i],page_text[j]) for j in pool}
   sortedpool.sort(key=lambda j:(-sims[j],-float(raw[i,j]),j))
  pred=[D[j]['id'] for j in sortedpool]
  m=per['methods'][name]
  assert m['candidate_pool']==[D[j]['id'] for j in pool]
  assert m['ordered_pages']==pred
  for point in (1,5):
   assert m[f'origin_top{point}']==int(origin in pred[:point])
   assert m[f'equiv_top{point}']==int(bool(gold&set(pred[:point])))
  assert m['origin_recall20']==int(origin in [D[j]['id'] for j in pool])
  assert m['equiv_recall20']==int(bool(gold & {D[j]['id'] for j in pool}))
print('verified independently all 140 x 3 ranked decisions, sums and gold hits',round(time.time()-start,1),flush=True)

# Independently regenerate final preregistered contrast and miss denominator.
rep=[p for p in P if p['anchor_type']=='repeated' and p['form']=='keyword6']
assert len(rep)==35
miss=[p for p in rep if not p['methods']['BM25']['origin_top1']]
recovered=[p for p in miss if p['methods']['BM25']['equiv_top1']]
hyp=R['hypotheses']
assert (len(recovered),len(miss))==(hyp['H23a']['numerator'],hyp['H23a']['denominator'])
assert abs(len(recovered)/len(miss)-hyp['H23a']['fraction'])<1e-12
for s in ('IMVO','ISIG'):
 z=[r for r in rep if r['source']==s]
 diff=np.mean([(r['methods']['BM25_FUZZY20']['equiv_top1']-r['methods']['BM25']['equiv_top1'])-(r['methods']['BM25_FUZZY20']['origin_top1']-r['methods']['BM25']['origin_top1']) for r in z])
 assert abs(diff-hyp['H23b']['by_source'][s])<1e-12
for p in P:
 if p['anchor_type']=='unique':
  for name,m in p['methods'].items():
   assert all(m[f'origin_{k}']==m[f'equiv_{k}'] for k in ('top1','top5','recall20'))
print('VERIFICATION PASS: gold labels, issue exclusions, SHA, independent BM25 scores, rankings, H23a/H23b/H23c',round(time.time()-start,1),flush=True)
