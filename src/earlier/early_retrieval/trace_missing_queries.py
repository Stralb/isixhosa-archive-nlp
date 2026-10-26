"""Qualitative provenance check: do absent language-view phrases stitch raw-separated fragments?"""
from pathlib import Path
import pickle,json,regex,collections
O=Path('/mnt/data/isiXhosa_three_track_pilot')
P=pickle.load(open(O/'corpus_cache.pkl','rb')); pages={r['id']:r for r in P}
q=[json.loads(s) for s in open(O/'queries.jsonl')]
v=json.loads((O/'query_exact_presence.json').read_text())
pat=regex.compile(r'(?V1)\p{L}+(?:[\'’]\p{L}+)*')
def words(s):return [x.lower() for x in pat.findall(s)]
def find_all(w,needle):
 return [i for i in range(len(w)-len(needle)+1) if w[i:i+len(needle)]==needle]
D=[]
for r in q:
 if v[r['id']]:continue
 raw=words(pages[r['page']]['raw']); seq=words(r['query'])
 examples=[]
 for split in range(2,len(seq)-1):
  left=find_all(raw,seq[:split]);right=find_all(raw,seq[split:])
  for a in left[:20]:
   for b in right[:20]:
    if b>a+split and b-(a+split)<600:
     examples.append(dict(split=split,missing_intervening_word_count=b-(a+split),raw_between=' '.join(raw[a+split:min(b,a+split+20)]),
                          left=' '.join(seq[:split]),right=' '.join(seq[split:]),page=r['page']))
 if examples:examples.sort(key=lambda a:(a['missing_intervening_word_count'],-min(a['split'],len(seq)-a['split'])))
 D.append(dict(id=r['id'],source=r['source'],lang=r['lang'],has_raw_fragment_trace=bool(examples),example=examples[0] if examples else None,
               old_query=r['query']))
with (O/'missing_query_trace.jsonl').open('w') as f:
 for row in D:f.write(json.dumps(row,ensure_ascii=False)+'\n')
print('absent',len(D),'reconstructable_as_stitch',sum(d['has_raw_fragment_trace'] for d in D),flush=True)
for a in D:
 if a['has_raw_fragment_trace']:
  print(json.dumps(a,ensure_ascii=False)[:500]);
  if sum(d['has_raw_fragment_trace'] for d in D[:D.index(a)+1])>=5:break
