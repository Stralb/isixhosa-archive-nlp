from pathlib import Path
import pickle,json,re,regex as rx,hashlib, numpy as np
from collections import defaultdict,Counter
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.linear_model import LogisticRegression
import scipy.sparse as sp
P=Path('/mnt/data/isiXhosa_phase17_deeper_questions');D=pickle.load(open('/mnt/data/isiXhosa_three_track_pilot/corpus_cache.pkl','rb'))
W=rx.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
def words(s):return [w.lower() for w in W.findall(s)]
def paras(s):return [words(p) for p in re.split(r'\n\s*\n',s) if p.strip()]
# Paired length test: same source page, same selected 10-word excerpt; 5/7 derive from middle of 10.
q=[]
for d in D:
 tot=len(words(d['en']))
 if not 10<=tot<50:continue
 valid={tuple(p[i:i+10]) for p in paras(d['raw']) for i in range(len(p)-9)}
 cand=[]
 for para in paras(d['en']):
  for i in range(len(para)-9):
   gram=tuple(para[i:i+10])
   if gram in valid and len(set(gram))>=6 and sum(map(len,gram))>=43:cand.append(gram)
 if not cand:continue
 loc=int(hashlib.sha256(('paired10'+d['id']).encode()).hexdigest()[:12],16)%len(cand)
 gram=cand[loc]
 for length,start in ((5,2),(7,1),(10,0)):
  probe=gram[start:start+length];q.append({'id':d['id']+'|'+str(length),'source':d['source'],'issue':d['issue'],'source_page':d['id'],'length':length,'text':' '.join(probe)})
print('paired pages',len(q)//3,'by source',dict(Counter(z['source'] for z in q if z['length']==5)),flush=True)
# Same historic silver-label query LID as prior, trained only early Imvo 60-word segments.
tr=[];lab=[]
for d in D:
 if d['source']!='IMVO' or not (1884<=d['year']<=1887):continue
 for lang in ('xh','en'):
  t=words(d[lang]);
  if len(t)<60:continue
  for st in [max(0,len(t)//4-30),max(0,3*len(t)//4-30) if len(t)>=250 else None]:
   if st is not None:tr.append(' '.join(t[st:st+60]));lab.append(int(lang=='xh'))
v=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,max_features=80000,sublinear_tf=True,dtype=np.float32)
model=LogisticRegression(C=2.,class_weight='balanced',solver='liblinear',max_iter=300,random_state=20261023).fit(v.fit_transform(tr),lab)
prob=model.predict_proba(v.transform([z['text'] for z in q]))[:,1]
for z,p in zip(q,prob):z['p_xho']=float(p);z['auto_xh']=bool(p>=.5);z['correct_en']=int(p<.5)
# Multiple valid paragraph sources for each query length, no artificial one-page relevance.
patterns=defaultdict(list)
for i,z in enumerate(q):patterns[(z['length'],tuple(z['text'].split()))].append(i)
g=[set() for x in q]
for j,d in enumerate(D):
 for para in paras(d['raw']):
  for k in (5,7,10):
   for i in range(len(para)-k+1):
    ids=patterns.get((k,tuple(para[i:i+k])),())
    for qid in ids:g[qid].add(j)
index={d['id']:i for i,d in enumerate(D)}
for i,z in enumerate(q):assert index[z['source_page']] in g[i]
# BM25 original full-index and two language-fields, same settings as before.
N=len(D);score={}
for field in ('raw','en','xh'):
 print('fit',field,flush=True)
 vect=CountVectorizer(lowercase=True,token_pattern=r'(?u)\b[^\W\d_]+\b',dtype=np.float32)
 X=vect.fit_transform([d[field] for d in D]).tocsr()
 length=np.asarray(X.sum(axis=1)).ravel().astype(np.float32);df=np.asarray((X>0).sum(axis=0)).ravel();idf=np.log1p((N-df+.5)/(df+.5));mu=float(length.mean())
 c=X.tocoo();w=c.data*2.2/(c.data+1.2*(.25+.75*length[c.row]/mu))*idf[c.col]
 Wm=sp.csr_matrix((w,(c.row,c.col)),shape=c.shape,dtype=np.float32)
 T=vect.transform([z['text'] for z in q]);T.data[:]=1
 score[field]=(T@Wm.T).toarray().astype(np.float32)
 del vect,X,Wm,T,c
s=score
s['oracle']=s['en'];s['auto']=np.where(np.array([z['auto_xh'] for z in q])[:,None],s['xh'],s['en'])
rows=[];summary={}
for m,mat in s.items():
 summary[m]={}
 for i,z in enumerate(q):
  top=np.argsort(-mat[i],kind='stable')[:10]
  item={**z,'method':m,'multi':int(len(g[i])>1),'num_exact_sources':len(g[i]),'any1':int(top[0] in g[i] and mat[i,top[0]]>0),'any5':int(any(v in g[i] and mat[i,v]>0 for v in top[:5]))}
  rows.append(item)
 for n in (5,7,10):
  summary[m][str(n)]={}
  for src in ('IMVO','ISIG'):
   a=[x for x in rows if x['method']==m and x['length']==n and x['source']==src]
   if a:summary[m][str(n)][src]={'n':len(a),'any1':float(np.mean([x['any1'] for x in a])),'any5':float(np.mean([x['any5'] for x in a])),'lid_en_recall':float(np.mean([x['correct_en'] for x in a])),'unique_frac':float(np.mean([1-x['multi'] for x in a]))}
 print(m,summary[m],flush=True)
(P/'minority_paired_length_results.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
(P/'minority_paired_length_summary.json').write_text(json.dumps(summary,indent=2))
