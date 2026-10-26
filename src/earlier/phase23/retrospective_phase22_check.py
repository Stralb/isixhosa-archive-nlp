"""POST-RESULT/EXPLORATORY: apply fixed Phase23 exact text rule to Phase22 source excerpts.
No claims of independent validation; Phase22 was already inspected before Phase23.
"""
import collections,json,re,csv
from pathlib import Path
import regex
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'phase23';P22=ROOT/'phase22'
W=regex.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
def words(s):return [x.lower() for x in W.findall(s)]
D=[json.loads(x) for x in (ROOT/'pages.jsonl').open()]
raw={p['id']:p['raw'] for p in D}
R=[json.loads(x) for x in (P22/'per_query.jsonl').open()]
pairs={}
for r in R:
 pairs.setdefault(r['page_id'],{})[r['mode']]=r
assert len(pairs)==8
byfirst=collections.defaultdict(dict)
for pg,d in pairs.items():
 c=words(d['clean']['text'])
 source_phrases={tuple(w[i:i+13]) for para in re.split(r'\n\s*\n',raw[pg]) for w in [words(para)] for i in range(len(w)-12)}
 candidates=[tuple(c[i:i+13]) for i in range(len(c)-12) if tuple(c[i:i+13]) in source_phrases]
 if candidates:
  z=candidates[len(candidates)//2];byfirst[z[0]][z]=pg;d['fixed_13_anchor']=z
 else:d['fixed_13_anchor']=None
found=collections.defaultdict(set)
for p in D:
 for para in re.split(r'\n\s*\n',p['raw']):
  w=words(para)
  for i in range(len(w)-12):
   m=byfirst.get(w[i])
   if m:
    pg=m.get(tuple(w[i:i+13]))
    if pg is not None:found[pg].add(p['id'])
metrics=collections.defaultdict(lambda:collections.Counter())
rows=[]
for pg,item in pairs.items():
 gold=found[pg]
 for mode in ('external','clean'):
  r=item[mode]
  for method,z in r['methods'].items():
   if not item['fixed_13_anchor']:continue
   pred=z['top1_page'];strict=int(pred==pg);equiv=int(pred in gold)
   metrics[(mode,method)]['strict']+=strict;metrics[(mode,method)]['equiv']+=equiv;metrics[(mode,method)]['n']+=1
   rows.append({'page_id':pg,'mode':mode,'method':method,'exact_13_anchor_found':True,'gold_count':len(gold),
        'strict_top1':strict,'equiv_top1':equiv,'top1_page':pred})
with (O/'phase22_retrospective_evidence.csv').open('w',newline='') as f:
 wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
summary={'paired_pages':len(pairs),'pages_with_exact_13_anchor':sum(bool(item['fixed_13_anchor']) for item in pairs.values()),
 'strict_vs_exact_text_by_query_method':{mode+'_'+meth:dict(v) for (mode,meth),v in metrics.items()},
 'classification':'POSTHOC_RETROSPECTIVE_NOT_A_PRECOMMITTED_TEST','raw_ocr_provenance_verified':False}
(O/'phase22_retrospective_evidence.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
