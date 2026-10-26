import collections,json,re
from pathlib import Path
import regex
ROOT=Path(__file__).resolve().parents[1]
W=regex.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
P=[json.loads(l) for l in (ROOT/'pages.jsonl').open()]
seen=set(); eligible_iss=set(); c=collections.Counter(); details=collections.defaultdict(list)
for p in P:
 iss=p['issue'];seen.add(iss)
 ps=[len(W.findall(s)) for s in re.split(r'\n\s*\n',p['raw'])]
 n=sum(50<=x<=350 for x in ps)
 c[p['source']+'_pages']+=1
 c[p['source']+'_eligible_paragraphs']+=n
 if n:
  eligible_iss.add(iss)
  details[p['source']].append(n)
by={s:{'total_issues':len({p['issue'] for p in P if p['source']==s}),
   'eligible_issues':len({p['issue'] for p in P if p['source']==s}&eligible_iss),
   'eligible_pages':len([p for p in P if p['source']==s and any(50<=len(W.findall(t))<=350 for t in re.split(r'\n\s*\n',p['raw']))]),
   'eligible_paras':c[s+'_eligible_paragraphs']} for s in ('IMVO','ISIG')}
print(json.dumps(by,indent=2))
(ROOT/'phase24/feasibility.json').write_text(json.dumps(by,indent=2))
