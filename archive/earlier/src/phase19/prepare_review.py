from pathlib import Path
import json, re, hashlib, random
P=Path(__file__).resolve().parent
q=[json.loads(x) for x in (P/'isiXhosa_phase18_evidence_routing/balanced_en_queries.jsonl').open()]
exclude={z['issue'] for z in q}
W=re.compile(r"[a-zA-Z]+(?:['-][a-zA-Z]+)*")
C=[]
for row in map(json.loads,(P/'pages.jsonl').open()):
 if row['issue'] in exclude or row['page']=='p001':continue
 if row['source']=='IMVO' and row['year']<1890:continue
 if row['source']=='ISIG' and row['year']>1888:continue
 blocks=[]
 for i,par in enumerate(re.split(r'\n\s*\n',row['en'])):
  par=re.sub(r'\s+',' ',par).strip()
  words=W.findall(par)
  if not 60<=len(words)<=330:continue
  if len(set(x.lower() for x in words))<45:continue
  # exclude table-like or highly repetitive snippets
  if sum(x[0].isupper() for x in words)<4: continue
  if len(re.findall(r'[.!?;:]',par))<2: continue
  blocks.append(dict(paragraph=par,index=i,n=len(words),uniq=len(set(x.lower() for x in words)),caps=sum(x[0].isupper() for x in words)))
 if not blocks:continue
 b=sorted(blocks,key=lambda v:(abs(v['n']-130),-v['caps'],v['index']))[0]
 C.append({k:row[k] for k in ('id','source','issue','year','page','date')} | b)
# At most one page per issue, hashed order independent from retrieval scores.
byissue={}
for c in C:
 issue=c['issue']
 if issue not in byissue or (abs(c['n']-130),c['id']) < (abs(byissue[issue]['n']-130),byissue[issue]['id']):byissue[issue]=c
by=dict(IMVO=[],ISIG=[])
for x in byissue.values(): by[x['source']].append(x)
for src in by:
 by[src].sort(key=lambda v:hashlib.sha256((v['issue']+'independent_query_review_oct24').encode()).hexdigest())
 # ISIG full list; Imvo keep 40
 by[src]=by[src] if src=='ISIG' else by[src][:40]
 (P/f'review_{src}.jsonl').write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in by[src]))
 print(src,len(by[src]),'sample years',[(x['year'],x['id']) for x in by[src][:8]])
with (P/'review_snippets.txt').open('w') as f:
 for src in by:
  print('\n',src,file=f)
  for i,p in enumerate(by[src]):
   print(f'[{src}-{i:02}] {p["id"]} paragraph={p["index"]} N={p["n"]}',file=f)
   print(p['paragraph'][:1450],file=f)
   print(file=f)
