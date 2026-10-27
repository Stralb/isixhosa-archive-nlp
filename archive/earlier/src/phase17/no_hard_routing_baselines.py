from pathlib import Path
import pickle,json, numpy as np
from collections import defaultdict
P=Path('/mnt/data/isiXhosa_phase17_deeper_questions');O=Path('/mnt/data/isiXhosa_phase15_source_grounded_search')
D=pickle.load(open('/mnt/data/isiXhosa_three_track_pilot/corpus_cache.pkl','rb'));idx={d['id']:i for i,d in enumerate(D)}
def fusion(slist,k=60):
 n=slist[0].shape[0];d=slist[0].shape[1];out=np.zeros((n,d),dtype=np.float32)
 for scores in slist:
  assert scores.shape==(n,d)
  rank=np.argsort(-scores,axis=1,kind='stable')
  weights=(1.0/(k+np.arange(1,d+1))).astype(np.float32)
  rows=np.arange(n)[:,None]
  out[rows,rank] += np.where(scores[rows,rank]>0,weights[None,:],0.)
 return out
samples=[('paragraph_10_clean',None),('paragraph_10_noisy',None),('minority_5word_en',None)]
results={};M=583
scores=np.load(O/'paragraph_scores.npz');ms=np.load(P/'minority_bm25_scores.npz')
allq=[json.loads(x) for x in open(O/'paragraph_queries.jsonl')]
minor=[v for v in [json.loads(x) for x in open(P/'language_probes_results.jsonl')] if v['kind']=='5words_English_minor']
old_gold={x['id']:{idx[a] for a in x['match_pages']} for x in allq}
# Retrieve exact 5word gold from results by rescanning page paragraphs - only needed for final rank scoring.
# Rebuild small gold map once for strict any-valid-page evaluation; no rank outcomes used.
import regex as rx,re
word=rx.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
phrase=defaultdict(list)
for i,q in enumerate(minor):phrase[tuple(q['text'].lower().split())].append(q['id'])
mg=defaultdict(set)
for j,d in enumerate(D):
 for para in re.split(r'\n\s*\n',d['raw']):
  w=[s.lower() for s in word.findall(para)]
  for k in range(len(w)-4):
   ids=phrase.get(tuple(w[k:k+5]),())
   for id in ids:mg[id].add(j)
for cohort in ('paragraph_10_clean','paragraph_10_noisy','minority_5word_en'):
 if cohort=='minority_5word_en':
  q=minor; x={k:ms[k] for k in ('raw','en','xh')};x['auto']=np.where(np.array([v['pred_xho'] for v in q])[:,None],x['xh'],x['en']);x['oracle']=x['en'];g=mg
 else:
  q=allq;off=0 if cohort.endswith('clean') else M
  x={k:scores[k][off:off+M] for k in ('raw','en','xh')};x['auto']=scores['fielded_auto'][off:off+M];x['oracle']=scores['fielded_oracle'][off:off+M];g=old_gold
 x['RRF_fields']=fusion([x['en'],x['xh']]);x['RRF_all']=fusion([x['raw'],x['en'],x['xh']])
 methods={};rows=[]
 for method,mat in x.items():
  vals=[]
  for i,qi in enumerate(q):
   rank=np.argsort(-mat[i],kind='stable');good=g[qi['id']]
   any1=int(int(rank[0]) in good and mat[i,rank[0]]>0)
   any5=int(any(int(z) in good and mat[i,z]>0 for z in rank[:5]))
   vals.append(dict(any1=any1,any5=any5,source=qi['source'],issue=qi['issue'],qid=qi['id']))
   rows.append(dict(cohort=cohort,method=method,source=qi['source'],issue=qi['issue'],qid=qi['id'],any1=any1,any5=any5))
  methods[method]={s:{'n':len(z),'any1':float(np.mean([a['any1'] for a in z])),'any5':float(np.mean([a['any5'] for a in z]))} for s,z in [('all',vals),('IMVO',[a for a in vals if a['source']=='IMVO']),('ISIG',[a for a in vals if a['source']=='ISIG'])]}
  print(cohort,method,methods[method]['all'],methods[method]['ISIG'],flush=True)
 # Compare union top1 of original each field, up to three results. This is a slate@3, NOT top1.
 slate=[]
 for i,qi in enumerate(q):
  top=[]
  for method in ('raw','en','xh'):
   row=x[method][i];loc=int(np.argmax(row))
   if row[loc]>0:top.append(loc)
  slate.append({'source':qi['source'],'hit':int(any(z in g[qi['id']] for z in top))})
 methods['slate_top1_three_fields']={s:{'n':len(z),'any_in_three':float(np.mean([a['hit'] for a in z]))} for s,z in [('all',slate),('IMVO',[a for a in slate if a['source']=='IMVO']),('ISIG',[a for a in slate if a['source']=='ISIG'])]}
 results[cohort]=methods
with (P/'fusion_results.jsonl').open('w') as f:
 for r in rows:f.write(json.dumps(r)+'\n')
(P/'fusion_summary.json').write_text(json.dumps(results,indent=2))
