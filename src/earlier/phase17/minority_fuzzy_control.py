from pathlib import Path
import pickle,json,numpy as np,re,regex as rx
from collections import defaultdict
from rapidfuzz.fuzz import partial_ratio
P=Path('/mnt/data/isiXhosa_phase17_deeper_questions')
D=pickle.load(open('/mnt/data/isiXhosa_three_track_pilot/corpus_cache.pkl','rb'));Q=[x for x in map(json.loads,open(P/'language_probes_results.jsonl')) if x['kind']=='5words_English_minor'];S=np.load(P/'minority_bm25_scores.npz')['raw'];N=len(D)
index={d['id']:i for i,d in enumerate(D)};assert S.shape==(len(Q),N)
W=rx.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
lookup=defaultdict(list)
for i,q in enumerate(Q):lookup[tuple(q['text'].split())].append(i)
g=[set() for q in Q]
for j,d in enumerate(D):
 for para in re.split(r'\n\s*\n',d['raw']):
  tokens=[w.lower() for w in W.findall(para)]
  for k in range(len(tokens)-4):
   for i in lookup.get(tuple(tokens[k:k+5]),()):g[i].add(j)
normalizer=re.compile(r'[^a-z0-9]');texts=[normalizer.sub('',d['raw'].lower()) for d in D]
rows=[]
for i,q in enumerate(Q):
 ranks=np.argsort(-S[i],kind='stable')[:20]
 cand=[int(k) for k in ranks if S[i,k]>0]
 phrase=normalizer.sub('',q['text'].lower())
 fuzzy=sorted([(partial_ratio(phrase,texts[j]),float(S[i,j]),j) for j in cand],key=lambda a:(-a[0],-a[1],a[2]))
 top=[z[2] for z in fuzzy]
 rows.append({'id':q['id'],'source':q['source'],'issue':q['issue'],'multi':int(len(g[i])>1),'any1':int(bool(top) and top[0] in g[i]),'any5':int(any(z in g[i] for z in top[:5])),'gold_in_top20':int(any(z in g[i] for z in cand))})
result={}
for s in ('IMVO','ISIG'):
 a=[x for x in rows if x['source']==s]
 result[s]={'n':len(a),'top1':float(np.mean([x['any1'] for x in a])),'top5':float(np.mean([x['any5'] for x in a])),'top20_ceiling':float(np.mean([x['gold_in_top20'] for x in a])),'unique_n':sum(1-x['multi'] for x in a),'unique_top1':float(np.mean([x['any1'] for x in a if not x['multi']]))}
result['all']={'n':len(rows),'top1':float(np.mean([x['any1'] for x in rows])),'top20_ceiling':float(np.mean([x['gold_in_top20'] for x in rows]))}
(P/'minority_fuzzy_results.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
(P/'minority_fuzzy_summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
