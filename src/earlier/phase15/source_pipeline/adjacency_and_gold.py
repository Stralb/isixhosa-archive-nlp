"""Corpus-wide 10-gram source-view adjacency audit and task gold repair."""
from pathlib import Path
from collections import defaultdict,Counter
import json,pickle,regex as rx,hashlib,random,time,statistics
O=Path('/mnt/data/isiXhosa_three_track_pilot');t0=time.time()
with (O/'corpus_cache.pkl').open('rb') as f:docs=pickle.load(f)
qs=[json.loads(x) for x in (O/'queries.jsonl').open()]
TOK=rx.compile(r'(?V1)\p{L}+(?:[\'’]\p{L}+)*')
B=1099511628211;MASK=(1<<64)-1;LEN=10;BN=pow(B,LEN,1<<64)
def words(s):return [x.lower() for x in TOK.findall(s)]
def window_hashes(toks):
 xs=[hash(x)&MASK for x in toks]
 if len(xs)<LEN:return []
 h=0; hs=[]
 for j,x in enumerate(xs):
  h=(h*B+x)&MASK
  if j>=LEN:h=(h-xs[j-LEN]*BN)&MASK
  if j>=LEN-1:hs.append(h)
 return hs
def query_hash(seq):
 h=0
 for x in seq:h=(h*B+(hash(x)&MASK))&MASK
 return h
byid={r['id']:r for r in docs}
query_by_page=defaultdict(list)
for q in qs:query_by_page[q['page']].append(q)
newq=[];audit=[]
for di,d in enumerate(docs):
 raw=words(d['raw']);rawhash=set(window_hashes(raw))
 counts={'id':d['id'],'issue':d['issue'],'source':d['source'],'year':d['year']}
 for lang in ('xh','en'):
  source=words(d[lang]); a=window_hashes(source)
  bad=sum(h not in rawhash for h in a)
  counts[f'{lang}_windows']=len(a);counts[f'{lang}_nonverbatim']=bad
 for q in query_by_page.get(d['id'],[]):
  lang=q['lang']; toks=words(d[lang]);orig=words(q['query'])
  origverbatim=(len(orig)==LEN and query_hash(orig) in rawhash)
  if origverbatim:use=orig
  else:
   arr=window_hashes(toks)
   candidates=list(range(len(arr)))
   rnd=random.Random(int(hashlib.sha256(('repair#'+q['id']+'20261023').encode()).hexdigest()[:16],16));rnd.shuffle(candidates)
   use=next((toks[j:j+LEN] for j in candidates if arr[j] in rawhash),None)
  if use is None:
   counts['no_valid_query']=counts.get('no_valid_query',0)+1
  else:
   z={**q,'query':' '.join(use),'repaired':not origverbatim,'original_verbatim':origverbatim}
   newq.append(z)
 audit.append(counts)
 if (di+1)%800==0:print('audit',di+1,'elapsed',round(time.time()-t0,1),flush=True)
from collections import defaultdict
summary={}
for src in ('IMVO','ISIG'):
 for lang in ('xh','en'):
  arr=[d for d in audit if d['source']==src and d[f'{lang}_windows']]
  total=sum(d[f'{lang}_windows'] for d in arr)
  total_bad=sum(d[f'{lang}_nonverbatim'] for d in arr)
  summary[f'{src}_{lang}']={
      'pages_with_ngrams':len(arr),'total_10grams':total,'not_verbatim_in_raw':total_bad,
      'rate':(total_bad/total if total else None),
      'median_page_nonverbatim_fraction':statistics.median(d[f'{lang}_nonverbatim']/d[f'{lang}_windows'] for d in arr) if arr else None,
      'pages_with_any_nonverbatim':sum(d[f'{lang}_nonverbatim']>0 for d in arr)}
print('SUMMARY',json.dumps(summary,indent=2),flush=True)
print('Fixed valid',len(newq),'original',len(qs),'repaired',sum(q['repaired'] for q in newq),flush=True)
# index all *corrected* exact 10-gram queries across all raw pages
newq.sort(key=lambda q:q['id'])
qbyhash=defaultdict(list)
for q in newq:
 qbyhash[query_hash(words(q['query']))].append(q['id'])
hits=defaultdict(set)
for di,d in enumerate(docs):
 raw=words(d['raw'])
 for h in set(window_hashes(raw)):
  for queryid in qbyhash.get(h,[]):hits[queryid].add(d['id'])
 if (di+1)%1200==0:print('Match',di+1,'elapsed',round(time.time()-t0,1),flush=True)
for q in newq:
 assert q['page'] in hits[q['id']],(q['id'],q['page'])
 q['num_exact_match_pages']=len(hits[q['id']]);q['num_exact_match_issues']=len({p.rsplit('_',1)[0] for p in hits[q['id']]})
 q['match_pages']=sorted(hits[q['id']]);
# Noise replays original noising function with same q id, fixed edits independent of results
for q in newq:
 toks=q['query'].split(); rng=random.Random(int(hashlib.sha256((q['id']+'#noise20261023').encode()).hexdigest()[:16],16))
 elig=[i for i,w in enumerate(toks) if len(w)>=4];rng.shuffle(elig)
 for j in elig[:max(1,round(len(toks)*.25))]:
  w=toks[j];pos=rng.randrange(1,len(w)-1);choices='aeinorsutkgmhbd';c=rng.choice([x for x in choices if x!=w[pos]]);toks[j]=w[:pos]+c+w[pos+1:]
 k=rng.randrange(1,len(toks)-1)
 if rng.randrange(2)==0 and len(toks[k])>5:
  pos=rng.randrange(2,len(toks[k])-1);toks[k:k+1]=[toks[k][:pos],toks[k][pos:]]
 else:toks[k:k+2]=[toks[k]+toks[k+1]]
 q['noisy_query']=' '.join(toks)
with (O/'query_repaired.jsonl').open('w') as f:
 for q in newq:f.write(json.dumps(q,ensure_ascii=False)+'\n')
with (O/'cross_view_adjacency_pages.jsonl').open('w') as f:
 for item in audit:f.write(json.dumps(item)+'\n')
summary['query_audit']={
 'original_n':len(qs),'repaired_n':len(newq),'original_invalid_n':sum(q['repaired'] for q in newq),
 'by_stratum': {f'{src}_{lang}':{'n':sum(q['source']==src and q['lang']==lang for q in newq),'repaired_n':sum(q['source']==src and q['lang']==lang and q['repaired'] for q in newq),
 'multi_page_exact_n':sum(q['source']==src and q['lang']==lang and q['num_exact_match_pages']>1 for q in newq),
 'multiple_issue_exact_n':sum(q['source']==src and q['lang']==lang and q['num_exact_match_issues']>1 for q in newq),
 'median_num_matching_pages':statistics.median([q['num_exact_match_pages'] for q in newq if q['source']==src and q['lang']==lang]) if any(q['source']==src and q['lang']==lang for q in newq) else None}
  for src in ('IMVO','ISIG') for lang in ('xh','en')}}
(O/'cross_view_adjacency_summary.json').write_text(json.dumps(summary,indent=2))
print('QUERY AUDIT',json.dumps(summary['query_audit'],indent=2),flush=True)
print('DONE seconds',round(time.time()-t0,1),flush=True)
