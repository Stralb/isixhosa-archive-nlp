from pathlib import Path
import json,pickle,random,hashlib,re,regex as rx,numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from rapidfuzz.fuzz import partial_ratio
O=Path('/mnt/data/isiXhosa_phase18_evidence_routing')
D=pickle.load(open('/mnt/data/isiXhosa_three_track_pilot/corpus_cache.pkl','rb'))
q=[*map(json.loads,open(O/'balanced_en_queries.jsonl')),*map(json.loads,open(O/'balanced_xh_queries.jsonl'))]
R=[*map(json.loads,open(O/'balanced_query_results.jsonl'))];assert len(q)==len(R)==932
src=np.array([z['source'] for z in R]);lang=np.array([z['view'] for z in R]);test=np.array([z['split']=='test' for z in R])
blocked={q[i]['issue'] for i in np.where(test)[0]}
W=rx.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
def words(s):return [w.lower() for w in W.findall(s)]
training=[];labels=[]
for d in D:
 if d['source']!='IMVO' or not (1884<=d['year']<=1887) or d['issue'] in blocked:continue
 for lg in ('en','xh'):
  ww=words(d[lg]);lab=int(lg=='xh')
  if len(ww)>=10:
   rng=random.Random(int(hashlib.sha256((d['id']+lg+'shortcontrol').encode()).hexdigest()[:12],16))
   for _ in range(2):
    st=rng.randrange(len(ww)-4);training.append(' '.join(ww[st:st+5]));labels.append(lab)
v=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,max_features=80000,sublinear_tf=True,dtype=np.float32)
model=LogisticRegression(C=2.,class_weight='balanced',solver='liblinear',max_iter=300,random_state=20261023).fit(v.fit_transform(training),labels)
# Golden pages in original clean raw paragraphs, for all five conditions.
pat=defaultdict(list)
for i,z in enumerate(q):pat[tuple(z['text'].lower().split())].append(i)
gold=[set() for _ in q]
for j,d in enumerate(D):
 for para in re.split(r'\n\s*\n',d['raw']):
  ww=words(para)
  for k in range(len(ww)-4):
   for i in pat.get(tuple(ww[k:k+5]),()):gold[i].add(j)
assert all(gold)
C={}
A=np.load('/mnt/data/isiXhosa_phase17_deeper_questions/minority_bm25_scores.npz');B=np.load(O/'balanced_xh_bm25_scores.npz')
C['clean']=([z['text'] for z in q], {f:np.concatenate([A[f],B[f]]) for f in ('raw','en','xh')})
E=np.load(O/'noisy_bm25_scores.npz');C['original_char_boundary']=([z['noisy'] for z in map(json.loads,open(O/'synthetic_noisy_queries.jsonl'))], {f:E[f] for f in ('raw','en','xh')})
V=[json.loads(z) for z in open(O/'new_noise_variants.jsonl')];S=np.load(O/'noise_sensitivity_scores.npz')
for k,fam in enumerate(['char_only','char_plus_boundary','two_char_plus_boundary']):
 sl=slice(k*932,(k+1)*932);C[fam]=([z['text'] for z in V[sl]],{f:S[f][sl] for f in ('raw','en','xh')})
normalizer=re.compile(r'[^a-z0-9]');pages=[normalizer.sub('',d['raw'].lower()) for d in D]
# Load existing gated and raw outcomes for comparison, match q ordering and labels.
prev=[json.loads(z) for z in open(O/'lid_disjoint_results.jsonl')]
prev_by={(z['cohort'],z['qid']):z for z in prev}
res={};allrows=[]
for cohort,(texts,scores) in C.items():
 pred=model.predict(v.transform(texts));got=np.zeros(len(q),dtype=int);oracle=np.zeros(len(q),dtype=int);means={}
 cleanq=[normalizer.sub('',t.lower()) for t in texts]
 for i,z in enumerate(q):
  this={}
  for field in set([('xh' if pred[i] else 'en'),lang[i]]):
   a=scores[field][i];cands=[int(j) for j in np.argsort(-a,kind='stable')[:20] if a[j]>0]
   ranks=sorted(cands,key=lambda j:(-partial_ratio(cleanq[i],pages[j]),-float(scores['raw'][i,j]),j))
   this[field]=int(bool(ranks) and ranks[0] in gold[i])
  got[i]=this['xh' if pred[i] else 'en'];oracle[i]=this[lang[i]]
  original=prev_by[(cohort,z['id'])]
  assert int(pred[i])==int(original['pred_xh'])
  allrows.append({'cohort':cohort,'qid':z['id'],'issue':z['issue'],'source':src[i],'view':lang[i],'split':'test' if test[i] else 'dev','hard_fuzzy':int(got[i]),'oracle_view_fuzzy':int(oracle[i]),'gated_fuzzy':int(original['gated']),'raw_fuzzy':int(original['raw'])})
  if (i+1)%300==0:print('scored',cohort,i+1,flush=True)
 raw=np.array([prev_by[(cohort,z['id'])]['raw'] for z in q]);gated=np.array([prev_by[(cohort,z['id'])]['gated'] for z in q])
 def macro(a):return float(np.mean([a[test&(src==s)&(lang==lg)].mean() for s in ('IMVO','ISIG') for lg in ('en','xh')]))
 res[cohort]={'test_macro':{'raw_fuzzy':macro(raw),'hard_fuzzy':macro(got),'gated_fuzzy':macro(gated),'oracle_view_fuzzy':macro(oracle)},'test_group':{s+'_'+lg:{'n':int(np.sum(test&(src==s)&(lang==lg))),'raw':float(raw[test&(src==s)&(lang==lg)].mean()),'hard':float(got[test&(src==s)&(lang==lg)].mean()),'gated':float(gated[test&(src==s)&(lang==lg)].mean()),'oracle':float(oracle[test&(src==s)&(lang==lg)].mean())} for s in ('IMVO','ISIG') for lg in ('en','xh')}}
 print(cohort,res[cohort],flush=True)
(O/'hard_fuzzy_results.jsonl').write_text(''.join(json.dumps(z)+'\n' for z in allrows))
(O/'HARD_FUZZY_RESULTS.json').write_text(json.dumps({'protocol':open(O/'hard_fuzzy_protocol.sha256').read().split()[0],'cohorts':res,'caveats':['Oracle is metadata view, not best attainable search result','Training labels are archive fields','Synthetic corruptions; copied source queries']},indent=2))
