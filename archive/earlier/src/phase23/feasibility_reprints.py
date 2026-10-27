"""Feasibility inventory of exact 13-word passage reuse only; never scores retrieval."""
import collections,hashlib,json,re,time
from pathlib import Path
import regex
R=Path(__file__).resolve().parents[1]
W=regex.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
def words(s):return [x.lower() for x in W.findall(s)]
prev=[]
for f in ['isiXhosa_phase18_evidence_routing/balanced_query_results.jsonl','phase19/queries_precommit.jsonl','phase20/queries_frozen.jsonl','phase21/queries_frozen.jsonl','phase22/queries_external_frozen.jsonl']:
 prev += [json.loads(x) for x in (R/f).open()]
excluded={q['issue'] for q in prev}
start=collections.defaultdict(dict); C=[];count=collections.Counter();t0=time.time()
for line in (R/'pages.jsonl').open():
 p=json.loads(line)
 if p['issue'] in excluded:continue
 raw13={tuple(words(para)[j:j+13]) for para in re.split(r'\n\s*\n',p['raw']) for j in range(max(0,len(words(para))-12))}
 for ix,para in enumerate(re.split(r'\n\s*\n',p['xh'])):
  w=words(para)
  if not 50<=len(w)<=350:continue
  starts=sorted(set([0,(len(w)-13)//4,(len(w)-13)//2,(3*(len(w)-13))//4,len(w)-13]))
  for pos in starts:
   z=tuple(w[pos:pos+13])
   if z not in raw13 or len(set(z))<8 or sum(len(t)>=4 for t in z)<6:continue
   i=len(C);C.append({'p':p['id'],'source':p['source'],'issue':p['issue'],'para_index':ix,'anchor':z})
   start[z[0]].setdefault(z,[]).append(i)
print('eligible candidate spans',len(C),'unique distinct anchors',sum(len(v) for v in start.values()),'first pass secs',round(time.time()-t0,1),flush=True)
found=collections.defaultdict(set)
for line in (R/'pages.jsonl').open():
 p=json.loads(line)
 for para in re.split(r'\n\s*\n',p['raw']):
  w=words(para)
  for j in range(len(w)-12):
   db=start.get(w[j])
   if db is None:continue
   for i in db.get(tuple(w[j:j+13]),[]):found[i].add(p['id'])
print('second pass',round(time.time()-t0,1),flush=True)
for source in ('IMVO','ISIG'):
 a=[(i,c) for i,c in enumerate(C) if c['source']==source];m=[(i,c) for i,c in a if len(found[i])>1]
 print(source,'eligible',len(a),'repeated candidate spans',len(m),'repeated issues',len(set(c['issue'] for i,c in m)),'max copies',max([len(found[i]) for i,c in m] or [0]), 'unique issues',len(set(c['issue'] for i,c in a)))
 print('sample repeated',[(c['p'],len(found[i]),' '.join(c['anchor'][:5])) for i,c in m[:4]])
