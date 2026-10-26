"""Pre-protocol, no-ranking source inventory (no retrieval outcomes inspected)."""
import collections,json,re
from pathlib import Path
import regex
ROOT=Path(__file__).resolve().parents[1]
W=regex.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
P=[]
for fn in ['isiXhosa_phase18_evidence_routing/balanced_query_results.jsonl','phase19/queries_precommit.jsonl','phase20/queries_frozen.jsonl','phase21/queries_frozen.jsonl','phase22/queries_external_frozen.jsonl']:
 P.extend(json.loads(x) for x in (ROOT/fn).open())
excl={r['issue'] for r in P}
issues=collections.defaultdict(lambda:[0,0,0]);pars=collections.defaultdict(int)
for s in (ROOT/'pages.jsonl').open():
 d=json.loads(s)
 if d['issue'] in excl:continue
 issues[(d['source'],d['issue'])][0]+=1
 n=0
 for p in re.split(r'\n\s*\n',d['xh']):
  t=W.findall(p)
  if 50<=len(t)<=350:
   n+=1
   pars[d['source']]+=1
 issues[(d['source'],d['issue'])][1]+=n
 issues[(d['source'],d['issue'])][2]+=int(n>0)
print('EXCLUDED',len(excl))
for src in ['IMVO','ISIG']:
 z=[(iid,v) for (s,iid),v in issues.items() if s==src]
 print(src,'unused',len(z),'with XH paras',sum(v[1]>0 for _,v in z),'with 2+',sum(v[1]>=2 for _,v in z),'paras',pars[src])
 print('first eligible', [i for i,v in z if v[1] > 0][:10])
