"""Phase 20 frozen deterministic query construction, no retrieval ranking or model scores."""
import collections, hashlib, json, random, re
from pathlib import Path
import regex as rx
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
R=Path(__file__).resolve().parents[1]; O=R/'phase20'
D=[json.loads(x) for x in (R/'pages.jsonl').open()]
P18=[json.loads(x) for x in (R/'isiXhosa_phase18_evidence_routing/balanced_query_results.jsonl').open()]
P19=[json.loads(x) for x in (R/'phase19/queries_precommit.jsonl').open()]
EXCLUDED={x['issue'] for x in P18+P19}
SEED=2026102420
W=rx.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
def words(s):return [w.lower() for w in W.findall(s)]
def h(s):return int(hashlib.sha256(('P20|'+str(SEED)+'|'+s).encode()).hexdigest()[:16],16)
def has_slice(seq,span):
 n=len(span)
 for i in range(len(seq)-n+1):
  if seq[i:i+n]==span:return True
 return False

def get_anchor(toks, rawset, salt):
 starts=sorted(range(len(toks)-12),key=lambda i:(abs(i-(len(toks)-13)//2), h(salt+'|anchor|'+str(i))))
 for i in starts:
  segment=toks[i:i+13]
  if tuple(segment) not in rawset:continue
  phrase=segment[4:9]
  if sum(len(t)>=4 for t in phrase)<2:continue
  if len(set(phrase))<4:continue
  return segment,phrase
 return None

def get_keyword(toks,view,salt):
 # Six spaced, distinctive words from the same original paragraph; not a fluent semantic paraphrase.
 stop=ENGLISH_STOP_WORDS if view=='en' else {'imvo','isigidimi','xosa','xhosa','abantu','kuba'}
 eligible=[i for i,t in enumerate(toks) if len(t)>=5 and t not in stop and t.isalpha()]
 eligible=[i for i in eligible if toks.count(toks[i])<=2]
 if len(eligible)<6:return None
 # Source-length-weighted, deterministic selection across six distinct paragraph zones.
 n=len(toks); edges=[int(n*i/6) for i in range(7)]
 selected=[]; used=set()
 for b in range(6):
  lo,hi=edges[b],edges[b+1]
  candidates=[i for i in eligible if lo<=i<hi and toks[i] not in used]
  if not candidates:return None
  # Prefer substantial surface words, reproducible tie-break by salted hash.
  candidates.sort(key=lambda i:(-min(len(toks[i]),14),h(salt+'|term|'+str(i))))
  pos=candidates[0]; selected.append(pos);used.add(toks[pos])
 q=' '.join(toks[i] for i in selected)
 qwords=words(q)
 for i in range(len(qwords)-2):
  if has_slice(toks,qwords[i:i+3]):return None
 return q

def make_noise(clean,salt):
 toks=clean.split();rr=random.Random(h(salt+'|noise'));valid=[i for i,x in enumerate(toks) if len(x)>=4]
 if len(valid)<2:return None
 rr.shuffle(valid)
 for i in valid[:2]:
  s=toks[i];j=rr.randrange(1,len(s)-1)
  repl=rr.choice([c for c in 'aeinorsutkgmhbd' if c!=s[j]])
  toks[i]=s[:j]+repl+s[j+1:]
 k=rr.randrange(1,len(toks)-1)
 if rr.randrange(2)==0 and len(toks[k])>5:
  j=rr.randrange(2,len(toks[k])-1);toks[k:k+1]=[toks[k][:j],toks[k][j:]]
 else:toks[k:k+2]=[toks[k]+toks[k+1]]
 return ' '.join(toks)

# Build eligible issue candidates before selecting any source issues. No ranker is invoked here.
by=collections.defaultdict(list)
for d in D:
 if d['issue'] in EXCLUDED:continue
 rawset=None
 for v in ('en','xh'):
  for parix,para in enumerate(re.split(r'\n\s*\n',d[v])):
   toks=words(para)
   if not 65<=len(toks)<=350:continue
   salt=d['id']+'|'+v+'|'+str(parix)
   if rawset is None:
    rawpars=[words(x) for x in re.split(r'\n\s*\n',d['raw'])]
    rawset={tuple(seq[k:k+13]) for seq in rawpars for k in range(len(seq)-12)}
   anchor=get_anchor(toks,rawset,salt)
   if anchor is None:continue
   key=get_keyword(toks,v,salt)
   if key is None:continue
   noisy=make_noise(' '.join(anchor[1]),salt)
   if noisy is None:continue
   if tuple(anchor[0]) not in rawset:raise AssertionError('missing paragraph evidence')
   by[(d['source'],v,d['issue'])].append(dict(id=d['id'],issue=d['issue'],source=d['source'],view=v,
     source_para_index=parix,anchor=' '.join(anchor[0]),keyword=key,clean=' '.join(anchor[1]),noisy=noisy,
     paragraph_word_count=len(toks),paragraph_text_sha256=hashlib.sha256(para.encode()).hexdigest()))

selection=[];selected_issues=set()
for source,v,count in (('ISIG','en',4),('IMVO','en',50),('ISIG','xh',50),('IMVO','xh',50)):
 issues=sorted([ii for s,w,ii in by if s==source and w==v and ii not in selected_issues],key=lambda ii:h(source+'|'+v+'|issue|'+ii))
 if len(issues)<count:raise RuntimeError(f'Insufficient new source issues {(source,v)}: {len(issues)} < {count}')
 for ii in issues[:count]:
  valid=by[(source,v,ii)]
  notfront=[p for p in valid if not p['id'].endswith('_p001.txt')]
  if notfront:valid=notfront
  valid=sorted(valid,key=lambda p:h('select|'+source+'|'+v+'|'+p['id']+'|'+str(p['source_para_index'])))
  selection.append(valid[0]);selected_issues.add(ii)

# All 3 query forms share the same strict evidence span, issue and page.
rows=[];audit=[]
for k,p in enumerate(selection):
 anchor=p['anchor'];qbase=f"P20_{p['source']}_{p['view'].upper()}_{k:03d}"
 for form,q in [('keyword',p['keyword']),('clean',p['clean']),('synthetic_noise',p['noisy'])]:
  rows.append({'qid':qbase+'_'+form,'anchor_id':qbase,'form':form,'query':q,'source':p['source'],
    'view':p['view'],'issue':p['issue'],'source_page':p['id'],'anchor_evidence_tokens':anchor,
    'query_origin': 'source_derived_nonverbatim_keyword_proxy' if form=='keyword' else
            ('literal_5word_excerpt' if form=='clean' else '2char_1boundary_synthetic_corruption')})
 audit.append({'anchor_id':qbase,**p})
assert len(selected_issues)==154 and len(selection)==154 and len(rows)==462
assert len(set(r['issue'] for r in rows))==154 and not selected_issues & EXCLUDED
# Three-token query overlaps are checked against the full originating paragraph during construction.
with (O/'queries_frozen.jsonl').open('w') as f:
 for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
with (O/'anchor_audit_frozen.jsonl').open('w') as f:
 for x in audit:f.write(json.dumps(x,ensure_ascii=False)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sumry={'source_view_counts':{f'{s}_{v}':sum(p['source']==s and p['view']==v for p in selection) for s in ('IMVO','ISIG') for v in ('en','xh')},
 'n_distinct_issues':len(selected_issues),'n_queries':len(rows),'excluded_issues':len(EXCLUDED),
 'sha256_queries':sha(O/'queries_frozen.jsonl'),'sha256_audit':sha(O/'anchor_audit_frozen.jsonl'),
 'genuine_independent_human_search':False,'genuine_ocr_evaluation':False,
 'first_12_anchor_queries':[{k:p[k] for k in ('id','view','keyword','clean','noisy')} for p in selection[:12]]}
(O/'construction_summary.json').write_text(json.dumps(sumry,indent=2,ensure_ascii=False))
print(json.dumps({k:v for k,v in sumry.items() if k!='first_12_anchor_queries'},indent=2))
print('query samples:',json.dumps(sumry['first_12_anchor_queries'][:6],indent=2))
