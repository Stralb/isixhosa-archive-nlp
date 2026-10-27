"""Phase 23 prospective query construction. NEVER invokes a retrieval model/score.

The exact-13-token multi-gold labels use paragraph-contained source text only.
"""
from __future__ import annotations
import collections,hashlib,json,re,time
from pathlib import Path
import regex
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'phase23';SEED=2026102423
W=regex.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
def words(s):return [x.lower() for x in W.findall(s)]
def hashhex(s):return hashlib.sha256((f'P23|{SEED}|'+s).encode('utf-8')).hexdigest()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(OUT/'PRECOMMIT.md')==(OUT/'PRECOMMIT_SHA256.txt').read_text().split()[0]
prior={}
for label,fn in [('phase18','isiXhosa_phase18_evidence_routing/balanced_query_results.jsonl'),('phase19','phase19/queries_precommit.jsonl'),('phase20','phase20/queries_frozen.jsonl'),('phase21','phase21/queries_frozen.jsonl'),('phase22','phase22/queries_external_frozen.jsonl')]:
 prior[label]={json.loads(x)['issue'] for x in (ROOT/fn).open()}
EXCLUDED=set.union(*prior.values())
D=[json.loads(x) for x in (ROOT/'pages.jsonl').open()];assert len(D)==3608
idx={p['id']:p for p in D};C=[];starts=collections.defaultdict(dict)

def forms(z):
 lit=[]
 for i in range(7):
  x=z[i:i+7]
  if len(set(x))<4:continue
  lit.append(((-len(set(x)),abs(i-3),i),x))
 if not lit:return None
 literal=' '.join(min(lit,key=lambda x:x[0])[1])
 eligible=sorted({t for t in z if len(t)>=4 and t.isalpha()},key=lambda t:(-len(t),t))
 if len(eligible)<6:return None
 picked=sorted(eligible[:6]); candidates=[picked, picked[1:]+picked[:1],list(reversed(picked))]
 for selected in candidates:
  if all(tuple(selected[k:k+3]) not in {tuple(z[j:j+3]) for j in range(len(z)-2)} for k in range(4)):
   return literal,' '.join(selected)
 return None

start=time.time()
for p in D:
 if p['issue'] in EXCLUDED:continue
 rawpar=[words(s) for s in re.split(r'\n\s*\n',p['raw'])]
 raw13={tuple(t[j:j+13]) for t in rawpar for j in range(len(t)-12)}
 for para_i,para in enumerate(re.split(r'\n\s*\n',p['xh'])):
  toks=words(para)
  if not 50<=len(toks)<=350:continue
  positions=sorted(set([0,(len(toks)-13)//4,(len(toks)-13)//2,3*(len(toks)-13)//4,len(toks)-13]))
  for pos in positions:
   z=tuple(toks[pos:pos+13])
   if z not in raw13 or len(set(z))<8 or sum(len(t)>=4 for t in z)<6:continue
   f=forms(z)
   if not f:continue
   i=len(C)
   C.append({'anchor':z,'source_page':p['id'],'source':p['source'],'issue':p['issue'],'page':p['page'],
     'para_idx':para_i,'pos':pos,'literal7':f[0],'keyword6':f[1]})
   starts[z[0]].setdefault(z,[]).append(i)
print('PHASE23 candidates after query-form rules',len(C),'elapsed',round(time.time()-start,1),flush=True)
occ=collections.defaultdict(set)
for p in D:
 for para in re.split(r'\n\s*\n',p['raw']):
  t=words(para)
  for j in range(len(t)-12):
   m=starts.get(t[j])
   if not m:continue
   for i in m.get(tuple(t[j:j+13]),[]):occ[i].add(p['id'])
print('Full-archive exact-label scan complete',round(time.time()-start,1),flush=True)
by_issue=collections.defaultdict(lambda:{'repeated':[],'unique':[]});total=collections.Counter();reasons=[]
for i,c in enumerate(C):
 pages=occ[i]
 assert c['source_page'] in pages,(i,c)
 iss={idx[p]['issue'] for p in pages}
 typ='repeated' if len(iss)>=2 else 'unique' if len(pages)==1 else None
 if typ:
  by_issue[(c['source'],c['issue'])][typ].append(i)
  total[(c['source'],typ)]+=1
else_type=0
chosen=[];chosen_issues=[]
for s,n in [('IMVO',28),('ISIG',7)]:
 eligible=sorted([iss for (src,iss),d in by_issue.items() if src==s and d['repeated'] and d['unique']],
                 key=lambda issue:hashhex('issue|'+s+'|'+issue))
 assert len(eligible)>=n,('not enough issues',s,len(eligible),n)
 for issue in eligible[:n]:
  pool=by_issue[(s,issue)]
  ri=min(pool['repeated'],key=lambda i:hashhex(f'repeat|{C[i]["source_page"]}|{C[i]["para_idx"]}|{C[i]["pos"]}'))
  rr=C[ri]
  def uniq_key(i):
   x=C[i]
   pref=0 if x['source_page']==rr['source_page'] and x['para_idx']==rr['para_idx'] else 1 if x['source_page']==rr['source_page'] else 2
   return (pref,hashhex(f'unique|{x["source_page"]}|{x["para_idx"]}|{x["pos"]}'))
  ui=min(pool['unique'],key=uniq_key)
  assert ri!=ui
  chosen.extend([('repeated',ri),('unique',ui)])
  chosen_issues.append(issue)
assert len(chosen)==70 and len(set(chosen_issues))==35 and not set(chosen_issues)&EXCLUDED

qrows=[];audit=[]
for ai,(typ,i) in enumerate(chosen):
 c=C[i];gold=sorted(occ[i]);gissue=len({idx[p]['issue'] for p in gold})
 assert (typ=='repeated' and gissue>=2) or (typ=='unique' and len(gold)==1)
 anchor_id=f'P23_{c["source"]}_{ai:03d}'
 rec={'anchor_id':anchor_id,'anchor_type':typ,'anchor':' '.join(c['anchor']),'anchor_tokens':list(c['anchor']),
      'source_page':c['source_page'],'source':c['source'],'issue':c['issue'],'source_para_index':c['para_idx'],
      'anchor_start':c['pos'],'gold_pages':gold,'exact_gold_count':len(gold),'exact_gold_issue_count':gissue,
      'query_literal7':c['literal7'],'query_keyword6':c['keyword6']}
 audit.append(rec)
 for form in ('literal7','keyword6'):
  qrows.append({'qid':f'{anchor_id}_{form}','anchor_id':anchor_id,'anchor_type':typ,'form':form,
    'query':c[form],'source':c['source'],'issue':c['issue'],'source_page':c['source_page'],
    'anchor_evidence_tokens':' '.join(c['anchor']),'gold_pages':gold,
    'query_origin':'source_derived_literal_7_word_passage' if form=='literal7' else 'source_derived_nonverbatim_keywords'})
for fn,rows in [('queries_frozen.jsonl',qrows),('anchor_audit_frozen.jsonl',audit)]:
 with (OUT/fn).open('w') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
summary={'n_raw_pages':len(D),'eligible_candidate_anchors':len(C),
 'unused_query_issues':{s:sum(k[0]==s for k in by_issue) for s in ('IMVO','ISIG')},
 'eligible_reprint_unique_paired_issues':{s:sum(k[0]==s and bool(v['repeated'] and v['unique']) for k,v in by_issue.items()) for s in ('IMVO','ISIG')},
 'sample_issues':dict(collections.Counter(r['source'] for r in audit[::2])),
 'sample_anchor_types':dict(collections.Counter(r['anchor_type'] for r in audit)),
 'gold_pages_distribution_repeated':dict(sorted(collections.Counter(r['exact_gold_count'] for r in audit if r['anchor_type']=='repeated').items())),
 'prior_origin_excluded_counts':{k:len(v) for k,v in prior.items()},'selected_issues':chosen_issues,
 'hashes':{fn:sha(OUT/fn) for fn in ('queries_frozen.jsonl','anchor_audit_frozen.jsonl','PRECOMMIT.md')},
 'no_retrieval_scoring_during_selection':True,'true_independent_queries':False,
 'first_two_anchor_examples':[{'issue':r['issue'],'type':r['anchor_type'],'query':r['query_keyword6']} for r in audit[:2]]}
(OUT/'construction_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k not in ('selected_issues','first_two_anchor_examples')},indent=2),flush=True)
print('CONSTRUCTION COMPLETE',round(time.time()-start,1),flush=True)
