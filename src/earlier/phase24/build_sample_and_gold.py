"""Precommitted, retrieval-score-blind construction of two UNENRICHED anchor frames.
Reads raw pages, never reads scores of any retrieval method.
"""
from __future__ import annotations
import collections,hashlib,json,re,time
from pathlib import Path
import numpy as np
import regex
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'phase24'; T=time.time()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(OUT/'PRECOMMIT.md')==(OUT/'PRECOMMIT_SHA256.txt').read_text().split()[0]
W=regex.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
def words(s):return [x.lower() for x in W.findall(s)]
def candidate(t,i):
 z=t[i:i+13]
 if len(z)!=13 or len(set(z))<8:return None
 words4={x for x in z if len(x)>=4 and x.isalpha()}
 if len(words4)<6:return None
 picked=sorted(sorted(words4,key=lambda x:(-len(x),x))[:6])
 if any(tuple(picked[k:k+3]) in {tuple(z[j:j+3]) for j in range(11)} for k in range(4)):
  return None
 return ' '.join(picked)
# Collect per-page paragraph eligible-start COUNTS, without inspecting other-page occurrence.
D=[json.loads(x) for x in (ROOT/'pages.jsonl').open()]; assert len(D)==3608
byissue=collections.defaultdict(list);counts=[];n_eligible=0
for j,p in enumerate(D):
 byissue[p['issue']].append(j)
 paragraphs=[words(t) for t in re.split(r'\n\s*\n',p['raw'])]
 meta=[]
 for k,t in enumerate(paragraphs):
  if not 50<=len(t)<=350:continue
  cnt=sum(candidate(t,i) is not None for i in range(len(t)-12))
  if cnt:
   meta.append({'index':k,'count':cnt})
   n_eligible+=cnt
 counts.append(meta)
 if (j+1)%900==0:print('page feasibility',j+1, 'seconds',round(time.time()-T,1), flush=True)
assert len(byissue)==718
# Sampling: one A, one B per issue. A: equal eligible pages, paragraphs, anchor positions.
# B: equal eligible start positions across issue.
rows=[];skipped=[]
for issue in sorted(byissue):
 page_ids=[j for j in byissue[issue] if counts[j]]
 if not page_ids:
  skipped.append(issue);continue
 for frame in ('A','B'):
  seed=int.from_bytes(hashlib.sha256(f'2026102424|{issue}|{frame}'.encode()).digest()[:8],'big')
  rng=np.random.default_rng(seed)
  if frame=='A':
   j=page_ids[int(rng.integers(len(page_ids)))]
   para=counts[j][int(rng.integers(len(counts[j])))]
   off=int(rng.integers(para['count']))
  else:
   sub=[(j,pm) for j in page_ids for pm in counts[j]]
   total=sum(pm['count'] for j,pm in sub)
   off=int(rng.integers(total))
   for j,pm in sub:
    if off<pm['count']:
     para=pm;break
    off-=pm['count']
  para_index=para['index']
  t=words(re.split(r'\n\s*\n',D[j]['raw'])[para_index])
  eligible=[(i,q) for i in range(len(t)-12) if (q:=candidate(t,i)) is not None]
  assert len(eligible)==para['count']
  pos,q=eligible[off]
  p=D[j]; a=t[pos:pos+13]
  rows.append({'qid':f'P24_{issue}_{frame}','frame':frame,'source':p['source'],'issue':issue,
    'page_id':p['id'],'paragraph_index':para_index,'anchor_start':pos,'anchor':' '.join(a),
    'anchor_tokens':a,'query':q,'query_source':'source_derived_six_keyword_nonverbatim',
    'sampling_seed':seed,'eligible_start_positions_in_paragraph':para['count']})
assert len(rows)==1436 and not skipped,(len(rows),skipped)
# Save sample and freeze it before constructing gold. No duplicate labels or scores accessed so far.
sample_path=OUT/'sample_frozen.jsonl'
with sample_path.open('w') as f:
 for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
(OUT/'SAMPLE_SHA256.txt').write_text(sha(sample_path)+'  sample_frozen.jsonl\n')
print('SAMPLE FROZEN', len(rows), 'seconds', round(time.time()-T,1),flush=True)
# Construct exact anchors' full-archive page/issue gold by complete single raw-paragraph scan.
starts=collections.defaultdict(set)
for r in rows:starts[r['anchor_tokens'][0]].add(tuple(r['anchor_tokens']))
occ=collections.defaultdict(set)
for j,p in enumerate(D):
 for par in re.split(r'\n\s*\n',p['raw']):
  t=words(par)
  for pos in range(len(t)-12):
   if t[pos] in starts:
    z=tuple(t[pos:pos+13])
    if z in starts[t[pos]]:occ[z].add(p['id'])
 if (j+1)%900==0:print('matching',j+1,'seconds',round(time.time()-T,1),flush=True)
page_issue={p['id']:p['issue'] for p in D}
for r in rows:
 pages=sorted(occ[tuple(r['anchor_tokens'])]);assert r['page_id'] in pages
 r['exact_gold_pages']=pages
 r['exact_gold_issue_count']=len({page_issue[p] for p in pages})
 r['exact_gold_page_count']=len(pages)
 r['cross_issue_repeat']=r['exact_gold_issue_count']>1
 r['within_issue_multipage_repeat']=len(pages)>1 and r['exact_gold_issue_count']==1
out=OUT/'queries_gold_frozen.jsonl'
with out.open('w') as f:
 for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
(OUT/'GOLD_SHA256.txt').write_text(sha(out)+'  queries_gold_frozen.jsonl\n')
previous={}
for label,fn in [('18','isiXhosa_phase18_evidence_routing/balanced_query_results.jsonl'),('19','phase19/queries_precommit.jsonl'),('20','phase20/queries_frozen.jsonl'),('21','phase21/queries_frozen.jsonl'),('22','phase22/queries_external_frozen.jsonl'),('23','phase23/queries_frozen.jsonl')]:
 previous[label]={json.loads(x)['issue'] for x in (ROOT/fn).open()}
exclude=set.union(*previous.values())
summary={'page_count':len(D),'issue_count':len(byissue),'total_eligible_anchor_positions':n_eligible,
 'n_frozen_rows':len(rows),'n_absent_issues':len(skipped),'source_counts':dict(collections.Counter(r['source'] for r in rows)),
 'sample_hash':sha(sample_path),'gold_hash':sha(out),'precommit_hash':sha(OUT/'PRECOMMIT.md'),
 'prior_issues_per_phase':{k:len(v) for k,v in previous.items()},
 'previously_unused_issues_by_source':{s:sorted(iss for iss in byissue if any(D[j]['source']==s for j in byissue[iss]) and iss not in exclude) for s in ('IMVO','ISIG')},
 'elapsed_seconds':round(time.time()-T,2)}
(OUT/'construction_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print('CONSTRUCTION COMPLETE',json.dumps({k:v for k,v in summary.items() if k!='previously_unused_issues_by_source'},indent=2),flush=True)
