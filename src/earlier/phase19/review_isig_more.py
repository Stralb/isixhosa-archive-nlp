from pathlib import Path
import json, re, hashlib
B=Path(__file__).resolve().parent
X={z['issue'] for z in map(json.loads,(B/'isiXhosa_phase18_evidence_routing/balanced_en_queries.jsonl').open())}
old={z['issue'] for z in map(json.loads,(B/'review_ISIG.jsonl').open())}
pages=[z for z in map(json.loads,(B/'pages.jsonl').open()) if z['source']=='ISIG' and z['issue'] not in X and z['page']!='p001' and z['issue'] not in old]
by={}
for p in pages:
 cand=[]
 for i,a in enumerate(re.split(r'\n\s*\n',p['en'])):
  para=re.sub(r'\s+',' ',a).strip();n=len(para.split())
  if n<30 or n>450:continue
  cand.append({'id':p['id'],'issue':p['issue'],'index':i,'n':n,'paragraph':para,'year':p['year'],'source':'ISIG'})
 if not cand:continue
 x=sorted(cand,key=lambda z:(abs(z['n']-100),z['id']))[0]
 if x['issue'] not in by or abs(x['n']-100) < abs(by[x['issue']]['n']-100):by[x['issue']]=x
rows=sorted(by.values(),key=lambda z:hashlib.sha256((z['issue']+'isig_extra').encode()).hexdigest());print(len(rows),'more issues')
with (B/'review_ISIG_extra.txt').open('w') as f:
 for i,z in enumerate(rows):
  print('EXTRA',i,z['id'],'N',z['n'],file=f)
  print(z['paragraph'][:900],file=f);print(file=f)
(B/'review_ISIG_extra.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
