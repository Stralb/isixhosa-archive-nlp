"""Phase 21: deterministic issue-disjoint source-view query construction.

No retrieval rank, score, vectorizer, language model or test-set performance is accessed.
"""
from __future__ import annotations
import collections, hashlib, json, random, re
from pathlib import Path
import regex as rx

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'phase21'; SEED=2026102421
D=[json.loads(x) for x in (ROOT/'pages.jsonl').open()]
P18=[json.loads(x) for x in (ROOT/'isiXhosa_phase18_evidence_routing/balanced_query_results.jsonl').open()]
P19=[json.loads(x) for x in (ROOT/'phase19/queries_precommit.jsonl').open()]
P20=[json.loads(x) for x in (ROOT/'phase20/queries_frozen.jsonl').open()]
EXCL={q['issue'] for q in P18+P19+P20}
assert len(D)==3608
W=rx.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
def words(s):return [w.lower() for w in W.findall(s)]
def digest_int(s):return int(hashlib.sha256(f'P21|{SEED}|{s}'.encode()).hexdigest()[:16],16)
def find_slice(tokens,needle):
 n=len(needle);return any(tokens[i:i+n]==needle for i in range(len(tokens)-n+1))
def anchor_from_paragraph(ptoks,raw13,salt):
 center=(len(ptoks)-13)//2
 for start in sorted(range(len(ptoks)-12),key=lambda i:(abs(i-center),digest_int(salt+'a'+str(i)))):
  segment=ptoks[start:start+13]
  if tuple(segment) not in raw13:continue
  clean=segment[4:9]
  if len(set(clean))<4 or sum(len(t)>=4 and t.isalpha() for t in clean)<3:continue
  return segment,clean
 return None

def keyword_from_paragraph(toks,salt):
 stop={'imvo','isigidimi','xosa','xhosa','abantu','kuba'} # identical transparent heuristic to Phase20
 cand=[i for i,t in enumerate(toks) if len(t)>=5 and t.isalpha() and t not in stop and toks.count(t)<=2]
 if len(cand)<6:return None
 n=len(toks);selected=[];used=set()
 for b in range(6):
  lo,hi=(n*b)//6,(n*(b+1))//6
  opts=[i for i in cand if lo<=i<hi and toks[i] not in used]
  if not opts:return None
  i=min(opts,key=lambda j:(-min(len(toks[j]),14),digest_int(salt+'k'+str(j))))
  selected.append(i);used.add(toks[i])
 q=[toks[i] for i in selected]
 if any(find_slice(toks,q[i:i+3]) for i in range(len(q)-2)):return None
 return ' '.join(q)

def corrupt(clean,salt,n_sub,boundary):
 toks=clean.split();rr=random.Random(digest_int(salt))
 eligible=[i for i,w in enumerate(toks) if len(w)>=4 and w.isalpha()]
 assert len(eligible)>=3 and n_sub<=3
 rr.shuffle(eligible)
 mutations=[]
 for i in eligible[:n_sub]:
  t=toks[i];j=rr.randrange(1,len(t)-1)
  options=[c for c in 'aeinorsutkgmhbd' if c!=t[j].lower()]
  new=rr.choice(options);toks[i]=t[:j]+new+t[j+1:]
  mutations.append({'word_pos':i,'char_pos':j,'before':t[j],'after':new})
 op=None
 if boundary:
  k=rr.randrange(1,len(toks)-1)
  if rr.randrange(2)==0 and len(toks[k])>5:
   j=rr.randrange(2,len(toks[k])-1);toks[k:k+1]=[toks[k][:j],toks[k][j:]]
   op={'type':'split','position':k,'char_pos':j}
  else:
   toks[k:k+2]=[toks[k]+toks[k+1]]
   op={'type':'join','position':k}
 return ' '.join(toks),{'mutations':mutations,'boundary':op}

# pre-score eligibility and deterministic issue selection
eligible=collections.defaultdict(list)
for d in D:
 if d['issue'] in EXCL:continue
 rawparas=[words(p) for p in re.split(r'\n\s*\n',d['raw'])]
 raw13={tuple(p[i:i+13]) for p in rawparas for i in range(len(p)-12)}
 if not raw13:continue
 for ip,p in enumerate(re.split(r'\n\s*\n',d['xh'])):
  toks=words(p)
  if not(65<=len(toks)<=350):continue
  salt=d['id']+'|xh|'+str(ip)
  anchor=anchor_from_paragraph(toks,raw13,salt)
  if anchor is None:continue
  key=keyword_from_paragraph(toks,salt)
  if key is None:continue
  spans,clean=anchor
  if not any(spans==rp[i:i+13] for rp in rawparas for i in range(len(rp)-12)):continue
  clean=' '.join(clean)
  noises={}
  for form,n_sub,boundary in [('edit1',1,False),('edit2_boundary',2,True),('edit3_boundary',3,True)]:
   noises[form]=corrupt(clean,salt+form,n_sub,boundary)
  if len({clean,key,*[v[0] for v in noises.values()]})!=5:continue
  eligible[(d['source'],d['issue'])].append({'source_page':d['id'],'source':d['source'],'issue':d['issue'],
     'source_paragraph_idx':ip,'source_paragraph_sha256':hashlib.sha256(p.encode()).hexdigest(),
     'source_paragraph_tokens':len(toks),'anchor':' '.join(spans),'clean':clean,'keyword':key,
     'noises':noises})

chosen=[];taken=set()
for source in ('IMVO','ISIG'):
 issues=sorted([k[1] for k in eligible if k[0]==source],key=lambda z:digest_int('issue|'+source+'|'+z))
 if len(issues)<30:raise RuntimeError('pre-ranking eligibility shortfall source '+source+': '+str(len(issues)))
 for ii in issues[:30]:
  opts=eligible[(source,ii)]
  nonfront=[p for p in opts if not p['source_page'].endswith('_p001.txt')]
  if nonfront:opts=nonfront
  chosen.append(min(opts,key=lambda x:digest_int('page|'+x['source_page']+'|'+str(x['source_paragraph_idx']))))
  taken.add(ii)
assert len(chosen)==60 and len(taken)==60 and not EXCL&taken

# Full-paragraph 13-gram equivalence gold on complete archive; determined solely from source text.
starts=collections.defaultdict(dict)
for i,c in enumerate(chosen):
 seq=tuple(words(c['anchor']));assert len(seq)==13
 starts[seq[0]].setdefault(seq,[]).append(i)
gold={i:set() for i in range(len(chosen))}
for d in D:
 for rawpar in re.split(r'\n\s*\n',d['raw']):
  toks=words(rawpar)
  for j,w in enumerate(toks):
   if w not in starts or j+13>len(toks):continue
   for ai in starts[w].get(tuple(toks[j:j+13]),[]):gold[ai].add(d['id'])
assert all(chosen[i]['source_page'] in gold[i] for i in range(len(chosen)))

rows=[];audit=[]
for i,c in enumerate(chosen):
 aid=f'P21_{c["source"]}_{i:03d}'
 for form in ('keyword','clean','edit1','edit2_boundary','edit3_boundary'):
  q=c[form] if form in ('keyword','clean') else c['noises'][form][0]
  assert len(words(q))>=3 and q
  rows.append({'qid':aid+'_'+form,'anchor_id':aid,'form':form,'query':q,'source':c['source'],
    'view':'xh','issue':c['issue'],'source_page':c['source_page'],'anchor_evidence_tokens':c['anchor'],
    'gold_pages':sorted(gold[i]),'query_origin':'archived_paragraph_proxy_or_synthetic_edit'})
 audit.append({k:v for k,v in c.items() if k!='noises'}|{'anchor_id':aid,'edit_ops':{k:v[1] for k,v in c['noises'].items()},
  'gold_count':len(gold[i]),'gold_pages':sorted(gold[i])})
assert len(rows)==300 and len({r['anchor_id'] for r in rows})==60

for name,records in [('queries_frozen.jsonl',rows),('anchor_audit.jsonl',audit)]:
 with (OUT/name).open('w') as f:
  for x in records:f.write(json.dumps(x,ensure_ascii=False)+'\n')

sha=lambda x:hashlib.sha256((OUT/x).read_bytes()).hexdigest()
summary={'selected_source_counts':dict(collections.Counter(c['source'] for c in chosen)),
  'eligible_source_issue_counts':{s:len([k for k in eligible if k[0]==s]) for s in ('IMVO','ISIG')},
  'queries':len(rows),'anchors':len(chosen),'query_file_sha256':sha('queries_frozen.jsonl'),
  'audit_file_sha256':sha('anchor_audit.jsonl'),
  'gold_multiplicity':dict(collections.Counter(len(g) for g in gold.values())),
  'all_source_issues_excluded_from_earlier_benchmarks':True,
  'independent_human_queries':False,'authentic_ocr':False}
(OUT/'construction_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2));print('first two anchors: ',json.dumps([{k:a[k] for k in ('source_page','keyword','clean')} for a in audit[:2]],ensure_ascii=False,indent=2))
