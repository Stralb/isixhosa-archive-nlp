from pathlib import Path
from collections import defaultdict,Counter
import re,regex as rx,json,pickle
import numpy as np
from rapidfuzz.fuzz import partial_ratio
O=Path('/mnt/data/isiXhosa_phase18_evidence_routing');P=Path('/mnt/data/isiXhosa_phase17_deeper_questions')
D=pickle.load(open('/mnt/data/isiXhosa_three_track_pilot/corpus_cache.pkl','rb'))
en=[json.loads(s) for s in open(O/'balanced_en_queries.jsonl')];xh=[json.loads(s) for s in open(O/'balanced_xh_queries.jsonl')];q=en+xh
R=[json.loads(s) for s in open(O/'balanced_query_results.jsonl')]
assert len(q)==len(R)==932
scores0=np.load(P/'minority_bm25_scores.npz');scores1=np.load(O/'balanced_xh_bm25_scores.npz');scores={field:np.concatenate([scores0[field],scores1[field]]) for field in ('raw','en','xh')}
word=rx.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
pat=defaultdict(list)
for i,z in enumerate(q):pat[tuple(z['text'].lower().split())].append(i)
gold=[set() for _ in q]
for j,d in enumerate(D):
 for para in re.split(r'\n\s*\n',d['raw']):
  tokens=[w.lower() for w in word.findall(para)]
  for t in range(len(tokens)-4):
   for i in pat.get(tuple(tokens[t:t+5]),()):gold[i].add(j)
assert all(gold)
# Pre-rank scores descending, stable; keep zero candidates out.
ranks={k:np.argsort(-mat,axis=1,kind='stable')[:,:50] for k,mat in scores.items()}
policies={'raw20':(20,0,0),'raw15_en5':(15,5,0),'raw15_xh5':(15,0,5),'raw10_en5_xh5':(10,5,5),'raw10_en10':(10,10,0),'raw10_xh10':(10,0,10),'raw5_en10_xh5':(5,10,5),'raw5_en5_xh10':(5,5,10)}
norm=re.compile(r'[^a-z0-9]');texts=[norm.sub('',d['raw'].lower()) for d in D]
Qnormal=[norm.sub('',z['text'].lower()) for z in q]
result={name:{'hit1':np.zeros(len(q),dtype=np.int8),'hit5':np.zeros(len(q),dtype=np.int8),'ceil20':np.zeros(len(q),dtype=np.int8),'pool_size':np.zeros(len(q),dtype=np.int8)} for name in policies}
# Cache fuzzy page match for query each page across policies.
for i,z in enumerate(q):
 hits={};candidates={}
 for name,alloc in policies.items():
  chosen=[];seen=set()
  for field,number in zip(('raw','en','xh'),alloc):
   if not number:continue
   valid=[int(j) for j in ranks[field][i] if scores[field][i,j]>0]
   for j in valid[:number]:
    if j not in seen:chosen.append(j);seen.add(j)
  for j in ranks['raw'][i]:
   j=int(j)
   if len(chosen)>=20:break
   if scores['raw'][i,j]>0 and j not in seen:chosen.append(j);seen.add(j)
  candidates[name]=chosen[:20]
  result[name]['pool_size'][i]=len(chosen[:20])
  result[name]['ceil20'][i]=int(any(j in gold[i] for j in chosen[:20]))
  for j in chosen[:20]:
   if j not in hits:hits[j]=partial_ratio(Qnormal[i],texts[j])
  # Same fuzzy scoring for every method, raw BM25 confidence tie-break.
  reranked=sorted(chosen[:20],key=lambda j:(-hits[j],-float(scores['raw'][i,j]),j))
  result[name]['hit1'][i]=int(bool(reranked) and reranked[0] in gold[i])
  result[name]['hit5'][i]=int(any(j in gold[i] for j in reranked[:5]))
 if (i+1)%100==0:print('scored',i+1,'of',len(q),flush=True)

source=np.array([z['source'] for z in q]);view=np.array(['en']*len(en)+['xh']*len(xh));partition=np.array([z['split'] for z in R]);unique=np.array([len(g)==1 for g in gold]);front=np.array([z['front'] for z in R]);

def macro(v,mask):
 a=[]
 for s in ('IMVO','ISIG'):
  for w in ('en','xh'):
   mm=mask & (source==s)&(view==w)
   if mm.sum():a.append(float(v[mm].mean()))
 return float(np.mean(a))

dev=partition=='dev';test=~dev
dev_scores={name:macro(v['hit1'],dev) for name,v in result.items()}
# Select best non-raw field pool only. Precommitted tie-break by simplest lexical representation order.
field_order=[k for k in policies if k!='raw20']
best=max(field_order,key=lambda k:(dev_scores[k],-field_order.index(k)))
print('dev scores',dev_scores,'selected',best)
summary={}
for setname,mask in [('dev',dev),('test',test)]:
 summary[setname]={}
 for name,valid in [('all',mask),('unique',mask&unique),('unique_nonfront',mask&unique&(~front)),('nonfront',mask&(~front))]+[(s+'_'+v,mask&(source==s)&(view==v)) for s in ('IMVO','ISIG') for v in ('en','xh')]:
  if valid.sum():summary[setname][name]={'n':int(valid.sum()),'methods':{k:{j:float(w[j][valid].mean()) for j in ('hit1','hit5','ceil20','pool_size')} for k,w in result.items()}}
 summary[setname]['macro']={'methods':{k:{j:macro(v[j],mask) for j in ('hit1','hit5','ceil20','pool_size')} for k,v in result.items()}}

# Paired issue bootstrap, shared issues for both views; resample issue clusters per newspaper.
rng=np.random.default_rng(20261023);B=10000
boot={}
for cohort,mask in [('test',test),('test_unique',test&unique),('test_nonfront',test&(~front))]:
 g={s:defaultdict(list) for s in ('IMVO','ISIG')}
 for i in np.where(mask)[0]:g[source[i]][q[i]['issue']].append(i)
 chunks={s:[np.array(a,dtype=int) for a in g[s].values()] for s in ('IMVO','ISIG')}
 choices={s:rng.integers(0,len(chunks[s]),size=(B,len(chunks[s]))) if chunks[s] else None for s in chunks}
 for method in [best]+field_order:
  if method!=best and cohort!='test':continue
  d=result[method]['hit1'].astype(float)-result['raw20']['hit1'].astype(float)
  distrib=[]
  for s in ('IMVO','ISIG'):
   for v in ('en','xh'):
    sums=np.array([d[ch][view[ch]==v].sum() for ch in chunks[s]])
    counts=np.array([(view[ch]==v).sum() for ch in chunks[s]])
    ids=choices[s]
    distrib.append(sums[ids].sum(axis=1)/np.maximum(counts[ids].sum(axis=1),1))
  dist=np.mean(distrib,axis=0)
  boot[cohort+'_'+method+'_minus_raw20']={'point':macro(d,mask),'ci95':list(map(float,np.quantile(dist,[.025,.975]))),'n':int(mask.sum())}
rows=[]
for i,z in enumerate(q):
 row={'qid':z['id'],'issue':z['issue'],'source':source[i],'view':view[i],'split':partition[i],'num_gold':len(gold[i]),'unique':bool(unique[i]),'front':bool(front[i])}
 for name,v in result.items():row[name+'_top1']=int(v['hit1'][i]);row[name+'_ceiling']=int(v['ceil20'][i])
 rows.append(row)
(O/'pool_results.jsonl').write_text(''.join(json.dumps(z)+'\n' for z in rows))
out={'sha256':open(O/'pool_protocol.sha256').read().split()[0],'candidate_policies':policies,'development_macro_top1':dev_scores,'development_selected_pool':best,'summary':summary,'bootstrap':boot,'caveats':['Both query fields from original index, not human search topics','All source golds are literal within-paragraph matches','Same partial_ratio fuzzy reranker on all candidate pools','Ties use RAW BM25 as in baseline']}
(O/'POOL_RESULTS.json').write_text(json.dumps(out,indent=2))
print('RESULTS')
for cohort in ('dev','test'):
 for group in ('macro','all','unique','unique_nonfront','IMVO_en','IMVO_xh','ISIG_en','ISIG_xh'):
  r=summary[cohort].get(group)
  if r:print(cohort,group,r.get('n'),{k:round(v['hit1']*100,1) for k,v in r['methods'].items()})
print('BOOT',boot)
