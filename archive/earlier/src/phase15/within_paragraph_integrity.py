"""Exact verification of view-paragraph words in source raw-paragraph text."""
import pickle,regex,re,json,collections,time
from pathlib import Path
O=Path('/mnt/data/isiXhosa_phase15_source_grounded_search');D=pickle.load(open('/mnt/data/isiXhosa_three_track_pilot/corpus_cache.pkl','rb'));P=regex.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
def words(x):return tuple(s.lower() for s in P.findall(x))
def paras(s):return [words(x) for x in re.split(r'\n\s*\n',s) if x.strip()]
def window(p,k=10):
 for j in range(len(p)-k+1):yield p[j:j+k]
A=collections.defaultdict(lambda:collections.Counter());bad=[];t=time.time()
for i,d in enumerate(D):
 rawset=set(q for para in paras(d['raw']) for q in window(para))
 for lang in ('xh','en'):
  group=d['source']+'_'+lang
  for par in paras(d[lang]):
   for g in window(par):
    A[group]['total']+=1
    if g not in rawset:
     A[group]['absent']+=1
     if len(bad)<20:bad.append((d['id'],lang,' '.join(g)))
 if (i+1)%800==0:print('pages',i+1,round(time.time()-t,1),flush=True)
S={k:dict(v) for k,v in A.items()};S['examples']=bad
(O/'within_paragraph_integrity_summary.json').write_text(json.dumps(S,indent=2,ensure_ascii=False))
print(json.dumps(S,indent=2,ensure_ascii=False),flush=True)
