from pathlib import Path
import json,pickle,hashlib,random,re,regex as rx,numpy as np,scipy.sparse as sp
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from rapidfuzz.fuzz import partial_ratio
O=Path('/mnt/data/isiXhosa_phase18_evidence_routing')
D=pickle.load(open('/mnt/data/isiXhosa_three_track_pilot/corpus_cache.pkl','rb'))
q=[*map(json.loads,open(O/'balanced_en_queries.jsonl')),*map(json.loads,open(O/'balanced_xh_queries.jsonl'))];r=[*map(json.loads,open(O/'balanced_query_results.jsonl'))]
FAMILIES=['char_only','char_plus_boundary','two_char_plus_boundary'];variants=[]
for fam in FAMILIES:
 for z in q:
  t=z['text'].split();rr=random.Random(int(hashlib.sha256(('PHASE18F|'+fam+'|'+z['id']).encode()).hexdigest()[:16],16))
  elig=[i for i,w in enumerate(t) if len(w)>=4];rr.shuffle(elig)
  n=2 if fam.startswith('two') else 1
  for i in elig[:n]:
   w=t[i];pos=rr.randrange(1,len(w)-1);c=rr.choice([v for v in 'aeinorsutkgmhbd' if v!=w[pos]]);t[i]=w[:pos]+c+w[pos+1:]
  if fam!='char_only':
   k=rr.randrange(1,len(t)-1)
   if rr.randrange(2)==0 and len(t[k])>5:
    pos=rr.randrange(2,len(t[k])-1);t[k:k+1]=[t[k][:pos],t[k][pos:]]
   else:t[k:k+2]=[t[k]+t[k+1]]
  variants.append(' '.join(t))
(O/'new_noise_variants.jsonl').write_text(''.join(json.dumps({'family':fam,'qid':q[i%len(q)]['id'],'text':variants[i]})+'\n' for i,fam in [(i,FAMILIES[i//len(q)]) for i in range(len(variants))]))
N=len(D);S={}
for field in ('raw','en','xh'):
 print('BM25',field,flush=True)
 v=CountVectorizer(lowercase=True,token_pattern=r'(?u)\b[^\W\d_]+\b',dtype=np.float32)
 X=v.fit_transform([d[field] for d in D]).tocsr();length=np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
 df=np.asarray((X>0).sum(axis=0)).ravel();idf=np.log1p((N-df+.5)/(df+.5));mu=max(float(length.mean()),1.0)
 co=X.tocoo();wt=co.data*2.2/(co.data+1.2*(.25+.75*length[co.row]/mu))*idf[co.col]
 Wt=sp.csr_matrix((wt,(co.row,co.col)),shape=co.shape,dtype=np.float32)
 T=v.transform(variants);T.data[:]=1
 S[field]=(T@Wt.T).toarray().astype(np.float32)
 del v,X,co,Wt,T,wt
np.savez_compressed(O/'noise_sensitivity_scores.npz',**S)
W=rx.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*");patterns=defaultdict(list)
for i,z in enumerate(q):patterns[tuple(z['text'].lower().split())].append(i)
gold=[set() for z in q]
for j,d in enumerate(D):
 for para in re.split(r'\n\s*\n',d['raw']):
  ww=[w.lower() for w in W.findall(para)]
  for k in range(len(ww)-4):
   for i in patterns.get(tuple(ww[k:k+5]),()):gold[i].add(j)
assert all(gold)
norm=re.compile(r'[^a-z0-9]');texts=[norm.sub('',d['raw'].lower()) for d in D];query=[norm.sub('',v.lower()) for v in variants]
rank={field:np.argsort(-score,axis=1,kind='stable')[:,:50] for field,score in S.items()}
pol={'raw20':(20,0,0),'raw15_en5':(15,5,0),'raw10_en5_xh5':(10,5,5)}
res={name:{'top1':np.zeros(len(variants),dtype=np.int8),'ceil20':np.zeros(len(variants),dtype=np.int8)} for name in pol}
for i in range(len(variants)):
 target=gold[i%len(q)];known={}
 for name,alloc in pol.items():
  pool=[];seen=set()
  for field,n in zip(('raw','en','xh'),alloc):
   if n:
    for j in [int(v) for v in rank[field][i] if S[field][i,v]>0][:n]:
     if j not in seen:pool.append(j);seen.add(j)
  for j in rank['raw'][i]:
   j=int(j)
   if len(pool)==20:break
   if S['raw'][i,j]>0 and j not in seen:pool.append(j);seen.add(j)
  pool=pool[:20]
  res[name]['ceil20'][i]=int(any(j in target for j in pool))
  for j in pool:
   if j not in known:known[j]=partial_ratio(query[i],texts[j])
  ordered=sorted(pool,key=lambda j:(-known[j],-float(S['raw'][i,j]),j))
  res[name]['top1'][i]=int(bool(ordered) and ordered[0] in target)
 if (i+1)%250==0:print('FUZZY',i+1,flush=True)
source=np.array([z['source'] for z in q]);view=np.array(['en']*466+['xh']*466);test=np.array([z['split']=='test' for z in r]);unique=np.array([len(g)==1 for g in gold]);front=np.array([z['front'] for z in r])
def macro(a,mask):
 return float(np.mean([a[mask&(source==s)&(view==v)].mean() for s in ('IMVO','ISIG') for v in ('en','xh')]))
rng=np.random.default_rng(20261023);B=10000
g={ss:defaultdict(list) for ss in ('IMVO','ISIG')}
for i in np.where(test)[0]:g[source[i]][q[i]['issue']].append(i)
chunks={ss:[np.array(z,dtype=int) for z in g[ss].values()] for ss in g}
choices={ss:rng.integers(0,len(chunks[ss]),size=(B,len(chunks[ss]))) for ss in chunks}
summary={};bootstrap={};rows=[];M=len(q)
for k,fam in enumerate(FAMILIES):
 sl=slice(k*M,(k+1)*M);rr={name:{met:v[met][sl] for met in ('top1','ceil20')} for name,v in res.items()}
 info={}
 for maskname,mask in [('test',test),('test_unique',test&unique),('test_nonfront',test&(~front))]:
  info[maskname]={'n':int(mask.sum()),'macro':{name:{met:macro(v[met],mask) for met in ('top1','ceil20')} for name,v in rr.items()},'by_group':{ss+'_'+vv:{name:float(v['top1'][mask&(source==ss)&(view==vv)].mean()) for name,v in rr.items()} for ss in ('IMVO','ISIG') for vv in ('en','xh')}}
 summary[fam]=info
 for method in ['raw15_en5','raw10_en5_xh5']:
  delta=rr[method]['top1'].astype(float)-rr['raw20']['top1'].astype(float)
  vec=[]
  for ss in ('IMVO','ISIG'):
   for vv in ('en','xh'):
    sums=np.array([delta[c][view[c]==vv].sum() for c in chunks[ss]])
    counts=np.array([(view[c]==vv).sum() for c in chunks[ss]])
    sample=choices[ss];vec.append(sums[sample].sum(axis=1)/np.maximum(counts[sample].sum(axis=1),1))
  dist=np.mean(vec,axis=0)
  bootstrap[fam+'_'+method+'_minus_raw20']={'point':macro(delta,test),'ci95':list(map(float,np.quantile(dist,[.025,.975])))}
 for i,z in enumerate(q):
  item={'qid':z['id'],'issue':z['issue'],'source':source[i],'view':view[i],'split':'test' if test[i] else 'dev','family':fam,'unique':bool(unique[i]),'front':bool(front[i])}
  for name,val in rr.items():item[name]=int(val['top1'][i]);item[name+'_ceiling']=int(val['ceil20'][i]);
  rows.append(item)
(O/'noise_sensitivity_results.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
(O/'NOISE_SENSITIVITY_RESULTS.json').write_text(json.dumps({'protocol':open(O/'sensitivity_protocol.sha256').read().split()[0],'families':FAMILIES,'summary':summary,'bootstrap':bootstrap},indent=2))
print('SUMMARIES')
for f in FAMILIES:
 print(f,'top1',{k:round(z['top1']*100,2) for k,z in summary[f]['test']['macro'].items()},'bootstrap',bootstrap[f+'_raw10_en5_xh5_minus_raw20'])
 print('by group',summary[f]['test']['by_group'])
