from pathlib import Path
import json,pickle,regex,re,numpy as np
from collections import defaultdict
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz.fuzz import partial_ratio
O=Path('/mnt/data/isiXhosa_phase18_evidence_routing')
D=pickle.load(open('/mnt/data/isiXhosa_three_track_pilot/corpus_cache.pkl','rb'))
q=[*map(json.loads,open(O/'balanced_en_queries.jsonl')),*map(json.loads,open(O/'balanced_xh_queries.jsonl'))]
prior=[*map(json.loads,open(O/'balanced_query_results.jsonl'))];assert len(q)==len(prior)==932
W=regex.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
@lru_cache(maxsize=200000)
def cg(w):
 s='^'+w.lower()+'$'
 return tuple(s[i:i+3] for i in range(len(s)-2))
def analyzer(text):
 for w in W.findall(text):yield from cg(w)
print('Fit char3 TFIDF',flush=True)
v=TfidfVectorizer(analyzer=analyzer,min_df=2,max_features=75000,sublinear_tf=True,dtype=np.float32)
X=v.fit_transform([d['raw'] for d in D]); print('FIT',X.shape,X.nnz,flush=True)
S=(v.transform([z['text'] for z in q])@X.T).toarray().astype(np.float32)
np.savez_compressed(O/'char_trigram_scores.npz',scores=S)
pat=defaultdict(list)
for i,z in enumerate(q):pat[tuple(z['text'].lower().split())].append(i)
gold=[set() for _ in q]
for j,d in enumerate(D):
 for para in re.split(r'\n\s*\n',d['raw']):
  tok=[w.lower() for w in W.findall(para)]
  for k in range(len(tok)-4):
   for i in pat.get(tuple(tok[k:k+5]),()):gold[i].add(j)
assert all(gold)
# Direct char3 then same top20 fuzzy. Source BM25 only for tie-break in both the old and new fuzzy baseline.
E=np.load('/mnt/data/isiXhosa_phase17_deeper_questions/minority_bm25_scores.npz')['raw']
F=np.load(O/'balanced_xh_bm25_scores.npz')['raw']
raw=np.concatenate([E,F]);norm=re.compile(r'[^a-z0-9]');docs=[norm.sub('',d['raw'].lower()) for d in D]
char=np.zeros(len(q),dtype=int);rerank=np.zeros(len(q),dtype=int);ceiling=np.zeros(len(q),dtype=int)
for i,z in enumerate(q):
 r=np.argsort(-S[i],kind='stable');char[i]=int(S[i,r[0]]>0 and int(r[0]) in gold[i])
 cand=[int(j) for j in r[:20] if S[i,j]>0]
 ceiling[i]=int(any(j in gold[i] for j in cand))
 query=norm.sub('',z['text'].lower())
 best=sorted(cand,key=lambda j:(-partial_ratio(query,docs[j]),-float(raw[i,j]),j))
 rerank[i]=int(bool(best) and best[0] in gold[i])
 if (i+1)%100==0:print('fuzzy',i+1,flush=True)
source=np.array([z['source'] for z in q]);view=np.array(['en']*466+['xh']*466);test=np.array([z['split']=='test' for z in prior]);unique=np.array([len(g)==1 for g in gold]);front=np.array([z['front'] for z in prior])
existing=[*map(json.loads,open(O/'pool_results.jsonl'))]
base=np.array([z['raw20_top1'] for z in existing]);pooled=np.array([z['raw10_en5_xh5_top1'] for z in existing]);

def macro(a,mask):
 return float(np.mean([a[mask&(source==ss)&(view==vv)].mean() for ss in ('IMVO','ISIG') for vv in ('en','xh')]))
summ={}
for cohort,mask in [('test',test),('test_unique',test&unique),('test_nonfront',test&(~front)),('dev',~test)]:
 summ[cohort]={'n':int(mask.sum()),'methods':{name:{'macro':macro(vec,mask),'micro':float(vec[mask].mean())} for name,vec in [('raw20_fuzzy',base),('pooled20_fuzzy',pooled),('char3',char),('char3_top20_fuzzy',rerank),('char3_candidate_ceiling',ceiling)]}}
 for ss in ('IMVO','ISIG'):
  for vv in ('en','xh'):
   z=mask&(source==ss)&(view==vv)
   summ[cohort][ss+'_'+vv]={'n':int(z.sum()),'methods':{name:float(vec[z].mean()) for name,vec in [('raw20_fuzzy',base),('pooled20_fuzzy',pooled),('char3',char),('char3_top20_fuzzy',rerank),('char3_candidate_ceiling',ceiling)]}}
rng=np.random.default_rng(20261023);B=10000
boot={};groups={s:defaultdict(list) for s in ('IMVO','ISIG')}
for i in np.where(test)[0]:groups[source[i]][q[i]['issue']].append(i)
chunks={s:[np.array(z,dtype=int) for z in groups[s].values()] for s in groups}
choices={s:rng.integers(0,len(chunks[s]),size=(B,len(chunks[s]))) for s in chunks}
for a,b,nm in [(rerank,base,'charfuzzy_minus_rawfuzzy'),(rerank,pooled,'charfuzzy_minus_poolfuzzy'),(pooled,base,'poolfuzzy_minus_rawfuzzy')]:
 dif=a.astype(float)-b.astype(float);dist=[]
 for s in ('IMVO','ISIG'):
  for vv in ('en','xh'):
   sums=np.array([dif[z][view[z]==vv].sum() for z in chunks[s]])
   counts=np.array([(view[z]==vv).sum() for z in chunks[s]])
   C=choices[s];dist.append(sums[C].sum(axis=1)/np.maximum(counts[C].sum(axis=1),1))
 m=np.mean(dist,axis=0)
 boot[nm]={'point':macro(dif,test),'ci95':list(map(float,np.quantile(m,[.025,.975])))}
(O/'CHAR_RESULTS.json').write_text(json.dumps({'methods':'TFIDF char3 and TFIDF char3+fuzzy top20, 75k vocabulary, exact Phase15 settings','protocol_sha256':open(O/'char_protocol.sha256').read().split()[0],'summary':summ,'bootstrap':boot},indent=2))
with (O/'char_results.jsonl').open('w') as f:
 for i,z in enumerate(q):f.write(json.dumps({'qid':z['id'],'split':'test' if test[i] else 'dev','source':source[i],'view':view[i],'unique':bool(unique[i]),'char_top1':int(char[i]),'char_fuzzy':int(rerank[i]),'char_candidate_hit':int(ceiling[i]),'raw_fuzzy':int(base[i]),'pool_fuzzy':int(pooled[i])})+'\n')
print(json.dumps({'summary':summ,'bootstrap':boot},indent=2),flush=True)
