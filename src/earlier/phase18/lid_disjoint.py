from pathlib import Path
import json,pickle,hashlib,random,regex as rx,numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
O=Path('/mnt/data/isiXhosa_phase18_evidence_routing')
D=pickle.load(open('/mnt/data/isiXhosa_three_track_pilot/corpus_cache.pkl','rb'))
q=[*map(json.loads,open(O/'balanced_en_queries.jsonl')),*map(json.loads,open(O/'balanced_xh_queries.jsonl'))];R=[*map(json.loads,open(O/'balanced_query_results.jsonl'))]
assert len(q)==len(R)==932
source=np.array([z['source'] for z in R]);view=np.array([z['view'] for z in R]);test=np.array([z['split']=='test' for z in R]);
blocked={q[i]['issue'] for i in np.where(test)[0]}
assert all(not q[i]['issue'] in blocked for i in np.where(~test)[0])
W=rx.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
def words(s):return [w.lower() for w in W.findall(s)]
training=[];labels=[];issues=set()
for d in D:
 if d['source']!='IMVO' or not (1884<=d['year']<=1887) or d['issue'] in blocked:continue
 issues.add(d['issue'])
 for lg in ('xh','en'):
  ww=words(d[lg]);flag=int(lg=='xh')
  if len(ww)>=10:
   rng=random.Random(int(hashlib.sha256((d['id']+lg+'shortcontrol').encode()).hexdigest()[:12],16))
   for _ in range(2):
    pos=rng.randrange(len(ww)-4);training.append(' '.join(ww[pos:pos+5]));labels.append(flag)
assert not issues.intersection(blocked)
print('LID fit examples',len(training),'training issues',len(issues),'blocked test issues',len(blocked),flush=True)
v=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,max_features=80000,sublinear_tf=True,dtype=np.float32)
model=LogisticRegression(C=2.,class_weight='balanced',solver='liblinear',max_iter=300,random_state=20261023).fit(v.fit_transform(training),labels)
cohorts={}
cohorts['clean']=[z['text'] for z in q]
cohorts['original_char_boundary']=[z['noisy'] for z in map(json.loads,open(O/'synthetic_noisy_queries.jsonl'))]
variants=[json.loads(z) for z in open(O/'new_noise_variants.jsonl')]
for k,fam in enumerate(['char_only','char_plus_boundary','two_char_plus_boundary']):cohorts[fam]=[z['text'] for z in variants[k*932:(k+1)*932]]
existing={}
for file_,key_,field_raw,field_en in [('pool_results.jsonl','clean','raw20_top1','raw15_en5_top1'),('noisy_query_results.jsonl','original_char_boundary','raw20_top1','raw15_en5_top1')]:
 t=[json.loads(s) for s in open(O/file_)];assert len(t)==932
 existing[key_]=(np.array([z[field_raw] for z in t]),np.array([z[field_en] for z in t]))
noisy=[json.loads(s) for s in open(O/'noise_sensitivity_results.jsonl')]
for k,fam in enumerate(['char_only','char_plus_boundary','two_char_plus_boundary']):
 t=noisy[k*932:(k+1)*932];existing[fam]=(np.array([z['raw20'] for z in t]),np.array([z['raw15_en5'] for z in t]))

def macro(vec,mask):return float(np.mean([vec[mask&(source==s)&(view==vv)].mean() for s in ('IMVO','ISIG') for vv in ('en','xh')]))

rng=np.random.default_rng(20261023);B=10000
issue={s:defaultdict(list) for s in ('IMVO','ISIG')}
for i in np.where(test)[0]:issue[source[i]][q[i]['issue']].append(i)
chunks={s:[np.array(z,dtype=int) for z in issue[s].values()] for s in issue};samples={s:rng.integers(0,len(chunks[s]),size=(B,len(chunks[s]))) for s in chunks}

def bootstrap(d):
 results=[]
 for s in ('IMVO','ISIG'):
  for lang in ('en','xh'):
   sums=np.array([d[g][view[g]==lang].sum() for g in chunks[s]])
   counts=np.array([(view[g]==lang).sum() for g in chunks[s]])
   ids=samples[s]
   results.append(sums[ids].sum(axis=1)/np.maximum(counts[ids].sum(axis=1),1))
 A=np.mean(results,axis=0)
 return {'point':macro(d,test),'ci95':list(map(float,np.quantile(A,[.025,.975])))}

summary={};raw_rows=[]
for cohort,queries in cohorts.items():
 x=model.predict(v.transform(queries));raw,en=existing[cohort]
 gated=np.where(x==0,en,raw)
 groups={s+'_'+lg:float(gated[test&(source==s)&(view==lg)].mean()-raw[test&(source==s)&(view==lg)].mean()) for s in ('IMVO','ISIG') for lg in ('en','xh')}
 summary[cohort]={'n_test':int(test.sum()),'test_macro_raw':macro(raw,test),'test_macro_gated':macro(gated,test),'gain_ci95':bootstrap(gated.astype(float)-raw.astype(float)),'by_group_gain':groups,'test_view_prediction_agreement':float((x[test]==np.array([1 if a=='xh' else 0 for a in view[test]])).mean()),'test_gate_english_fraction':float((x[test]==0).mean()),'xh_harmed_queries':int(np.sum((gated<raw)&test&(view=='xh')))}
 print(cohort,summary[cohort],flush=True)
 for i,z in enumerate(q):raw_rows.append({'cohort':cohort,'qid':z['id'],'issue':z['issue'],'source':source[i],'view':view[i],'split':'test' if test[i] else 'dev','pred_xh':int(x[i]),'raw':int(raw[i]),'gated':int(gated[i])})
(O/'ISSUE_DISJOINT_RESULTS.json').write_text(json.dumps({'protocol':open(O/'disjoint_protocol.sha256').read().split()[0],'n_training_issues':len(issues),'n_blocked_test_issues':len(blocked),'n_examples':len(training),'summary':summary,'limitations':['No expert language labels','Training still uses same newspaper corpus source across issues','Source copied query strings, not user questions','Synthetic noise only']},indent=2))
(O/'lid_disjoint_results.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in raw_rows))
