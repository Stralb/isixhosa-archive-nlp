from pathlib import Path
import re,regex as rx,json,pickle,random,hashlib,numpy as np,scipy.sparse as sp
from collections import defaultdict
from rapidfuzz.fuzz import partial_ratio
from sklearn.feature_extraction.text import CountVectorizer
O=Path('/mnt/data/isiXhosa_phase18_evidence_routing')
D=pickle.load(open('/mnt/data/isiXhosa_three_track_pilot/corpus_cache.pkl','rb'))
q=[*map(json.loads,open(O/'balanced_en_queries.jsonl')),*map(json.loads,open(O/'balanced_xh_queries.jsonl'))]
r=[*map(json.loads,open(O/'balanced_query_results.jsonl'))]
assert len(q)==len(r)==932
noisy=[]
for z in q:
 tok=z['text'].split()
 rnd=random.Random(int(hashlib.sha256((z['id']+'#noise20261023').encode()).hexdigest()[:16],16))
 elig=[i for i,w in enumerate(tok) if len(w)>=4];rnd.shuffle(elig)
 for j in elig[:max(1,round(len(tok)*.25))]:
  w=tok[j];pos=rnd.randrange(1,len(w)-1)
  c=rnd.choice([v for v in 'aeinorsutkgmhbd' if v!=w[pos]])
  tok[j]=w[:pos]+c+w[pos+1:]
 k=rnd.randrange(1,len(tok)-1)
 if rnd.randrange(2)==0 and len(tok[k])>5:
  pos=rnd.randrange(2,len(tok[k])-1);tok[k:k+1]=[tok[k][:pos],tok[k][pos:]]
 else:tok[k:k+2]=[tok[k]+tok[k+1]]
 noisy.append(' '.join(tok))
(O/'synthetic_noisy_queries.jsonl').write_text(''.join(json.dumps({'qid':q[i]['id'],'clean':q[i]['text'],'noisy':noisy[i]})+'\n' for i in range(len(q))))
# Score identical BM25 indexes for 932 noisy queries. All three independently fitted.
S={};N=len(D)
for field in ('raw','en','xh'):
 print('score field',field,flush=True)
 v=CountVectorizer(lowercase=True,token_pattern=r'(?u)\b[^\W\d_]+\b',dtype=np.float32)
 X=v.fit_transform([d[field] for d in D]).tocsr()
 lens=np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
 df=np.asarray((X>0).sum(axis=0)).ravel();idf=np.log1p((N-df+.5)/(df+.5));mu=max(float(lens.mean()),1)
 co=X.tocoo();weight=co.data*2.2/(co.data+1.2*(.25+.75*lens[co.row]/mu))*idf[co.col]
 Wgt=sp.csr_matrix((weight,(co.row,co.col)),shape=co.shape,dtype=np.float32)
 T=v.transform(noisy);T.data[:]=1
 S[field]=(T@Wgt.T).toarray().astype(np.float32)
 del v,X,co,Wgt,T,weight
np.savez_compressed(O/'noisy_bm25_scores.npz',**S)

# Gold is original text exact raw paragraph, independent of synthetic corruption.
W=rx.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*");lookup=defaultdict(list)
for i,z in enumerate(q):lookup[tuple(z['text'].lower().split())].append(i)
gold=[set() for _ in q]
for j,d in enumerate(D):
 for para in re.split(r'\n\s*\n',d['raw']):
  tok=[t.lower() for t in W.findall(para)]
  for i in range(len(tok)-4):
   for a in lookup.get(tuple(tok[i:i+5]),()):gold[a].add(j)
assert all(gold)
ranks={k:np.argsort(-mat,axis=1,kind='stable')[:,:50] for k,mat in S.items()}
policies={'raw20':(20,0,0),'raw15_en5':(15,5,0),'raw10_en5_xh5':(10,5,5)}
norm=re.compile(r'[^a-z0-9]');texts=[norm.sub('',d['raw'].lower()) for d in D];qn=[norm.sub('',w.lower()) for w in noisy]
R={p:{'top1':np.zeros(len(q),dtype=np.int8),'top5':np.zeros(len(q),dtype=np.int8),'ceil20':np.zeros(len(q),dtype=np.int8)} for p in policies}
for i,z in enumerate(q):
 fuzzycache={}
 for name,alloc in policies.items():
  chosen=[];seen=set()
  for field,k in zip(('raw','en','xh'),alloc):
   for j in [int(v) for v in ranks[field][i] if S[field][i,v]>0][:k]:
    if j not in seen:chosen.append(j);seen.add(j)
  for j in ranks['raw'][i]:
   j=int(j)
   if len(chosen)>=20:break
   if S['raw'][i,j]>0 and j not in seen:chosen.append(j);seen.add(j)
  chosen=chosen[:20]
  R[name]['ceil20'][i]=int(any(j in gold[i] for j in chosen))
  for j in chosen:
   if j not in fuzzycache:fuzzycache[j]=partial_ratio(qn[i],texts[j])
  ranks2=sorted(chosen,key=lambda j:(-fuzzycache[j],-float(S['raw'][i,j]),j))
  R[name]['top1'][i]=int(bool(ranks2) and ranks2[0] in gold[i]);R[name]['top5'][i]=int(any(j in gold[i] for j in ranks2[:5]))
 if (i+1)%120==0:print('noise fuzzy',i+1,flush=True)
source=np.array([z['source'] for z in q]);view=np.array(['en']*466+['xh']*466);test=np.array([z['split']=='test' for z in r]);unique=np.array([len(g)==1 for g in gold]);front=np.array([z['front'] for z in r]);
def macro(x,mask):return float(np.mean([x[mask&(source==s)&(view==v)].mean() for s in ('IMVO','ISIG') for v in ('en','xh')]))
summary={}
for cohort,mask in [('test',test),('dev',~test),('test_unique',test&unique),('test_nonfront',test&(~front))]:
 summary[cohort]={'n':int(mask.sum()),'macro':{name:{m:macro(x[m],mask) for m in ('top1','top5','ceil20')} for name,x in R.items()},'by_source_view':{s+'_'+v:{name:float(x['top1'][mask&(source==s)&(view==v)].mean()) for name,x in R.items()} for s in ('IMVO','ISIG') for v in ('en','xh')}}
rng=np.random.default_rng(20261023);B=10000
boot={}
chunks={s:defaultdict(list) for s in ('IMVO','ISIG')}
for i in np.where(test)[0]:chunks[source[i]][q[i]['issue']].append(i)
cl={s:[np.array(z,dtype=int) for z in chunks[s].values()] for s in chunks}
resamples={s:rng.integers(0,len(cl[s]),size=(B,len(cl[s]))) for s in cl}
for method in ('raw15_en5','raw10_en5_xh5'):
 for metric in ('top1','ceil20'):
  delta=R[method][metric].astype(float)-R['raw20'][metric].astype(float)
  buckets=[]
  for s in ('IMVO','ISIG'):
   for v in ('en','xh'):
    sums=np.array([delta[c][view[c]==v].sum() for c in cl[s]])
    counts=np.array([(view[c]==v).sum() for c in cl[s]])
    choice=resamples[s]
    buckets.append(sums[choice].sum(axis=1)/np.maximum(counts[choice].sum(axis=1),1))
  dist=np.mean(buckets,axis=0)
  boot[method+'_minus_raw20_'+metric]={'point':macro(delta,test),'ci95':list(map(float,np.quantile(dist,[.025,.975])))}
with (O/'noisy_query_results.jsonl').open('w') as f:
 for i,z in enumerate(q):
  rr={'qid':z['id'],'source':source[i],'view':view[i],'issue':z['issue'],'split':'test' if test[i] else 'dev','unique':bool(unique[i]),'noisy':noisy[i]}
  for name,val in R.items():rr[name+'_top1']=int(val['top1'][i]);rr[name+'_ceil20']=int(val['ceil20'][i]);
  f.write(json.dumps(rr)+'\n')
out={'sha256':open(O/'noise_protocol.sha256').read().split()[0],'summary':summary,'bootstrap':boot,'noisy_query_recipe':'Phase15 deterministic 25percent words, char substitution plus split or join','caveats':['OCR is synthetic; error doses are not OCR accuracy estimates','Gold is original clean source paragraph; annotation not independently human','Methods held frozen from clean development selection']}
(O/'NOISY_RESULTS.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'summary':summary,'boot':boot},indent=2))
