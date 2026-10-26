from pathlib import Path
import json,pickle,hashlib,random,regex as rx,numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
P=Path('/mnt/data/isiXhosa_phase18_evidence_routing')
Q=[*map(json.loads,open(P/'balanced_en_queries.jsonl')),*map(json.loads,open(P/'balanced_xh_queries.jsonl'))]
R=[*map(json.loads,open(P/'balanced_query_results.jsonl'))];assert len(R)==len(Q)==932
source=np.array([x['source'] for x in R]);view=np.array([x['view'] for x in R]);test=np.array([x['split']=='test' for x in R]);train=~test;unique=np.array([x['unique'] for x in R]);front=np.array([x['front'] for x in R]);

def macro(vec,mask):return float(np.mean([vec[mask&(source==s)&(view==v)].mean() for s in ('IMVO','ISIG') for v in ('en','xh')]))
# Fit previously specified short-query char ngram LID exactly as Phase17.
D=pickle.load(open('/mnt/data/isiXhosa_three_track_pilot/corpus_cache.pkl','rb'))
W=rx.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
def words(s):return [w.lower() for w in W.findall(s)]
texts=[];labels=[]
for d in D:
 if d['source']!='IMVO' or not (1884<=d['year']<=1887):continue
 for lg in ('xh','en'):
  ww=words(d[lg]);lab=int(lg=='xh')
  if len(ww)>=10:
   rng=random.Random(int(hashlib.sha256((d['id']+lg+'shortcontrol').encode()).hexdigest()[:12],16))
   for _ in range(2):
    st=rng.randrange(len(ww)-4);texts.append(' '.join(ww[st:st+5]));labels.append(lab)
v=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,max_features=80000,sublinear_tf=True,dtype=np.float32)
model=LogisticRegression(C=2.,class_weight='balanced',solver='liblinear',max_iter=300,random_state=20261023).fit(v.fit_transform(texts),labels)
clean_lid=model.predict(v.transform([z['text'] for z in Q]))
assert np.array_equal(clean_lid,np.array([x['lid_short_prediction_view']=='xh' for x in R],dtype=int))

# Fixed clean issue-development ratio gate.
K=np.load('/mnt/data/isiXhosa_phase17_deeper_questions/minority_bm25_scores.npz');Kx=np.load(P/'balanced_xh_bm25_scores.npz');
en=np.concatenate([K['en'],Kx['en']]);xh=np.concatenate([K['xh'],Kx['xh']]);ratio=en.max(axis=1)/(xh.max(axis=1)+1e-9)
clean_raw=np.array([z['raw20_top1'] for z in map(json.loads,open(P/'pool_results.jsonl'))]);clean_en=np.array([z['raw15_en5_top1'] for z in map(json.loads,open(P/'pool_results.jsonl'))]);
thetas=[0.75,1.,1.25,1.5,2.,3.]
theta_dev={str(t):macro(np.where(ratio>=t,clean_en,clean_raw),train) for t in thetas}
theta=max(thetas,key=lambda t:(theta_dev[str(t)],t))
print('Clean dev threshold',theta,theta_dev,flush=True)

# Cohorts: clean, original Phase18E noise, Phase18F A/B/C. Build exactly same q order.
cohorts={'clean':( [z['text'] for z in Q],ratio,clean_raw,clean_en)}
N=[json.loads(s) for s in open(P/'synthetic_noisy_queries.jsonl')]
S=np.load(P/'noisy_bm25_scores.npz');cohorts['original_char_boundary']=([z['noisy'] for z in N],S['en'].max(axis=1)/(S['xh'].max(axis=1)+1e-9),np.array([z['raw20_top1'] for z in map(json.loads,open(P/'noisy_query_results.jsonl'))]),np.array([z['raw15_en5_top1'] for z in map(json.loads,open(P/'noisy_query_results.jsonl'))]))
ns=[json.loads(s) for s in open(P/'new_noise_variants.jsonl')];rr=[json.loads(s) for s in open(P/'noise_sensitivity_results.jsonl')];T=np.load(P/'noise_sensitivity_scores.npz');assert len(ns)==len(rr)==3*len(Q)
for ki,fam in enumerate(['char_only','char_plus_boundary','two_char_plus_boundary']):
 sl=slice(ki*len(Q),(ki+1)*len(Q));
 assert all(z['family']==fam for z in ns[sl])
 cohorts[fam]=([z['text'] for z in ns[sl]],T['en'][sl].max(axis=1)/(T['xh'][sl].max(axis=1)+1e-9),np.array([z['raw20'] for z in rr[sl]]),np.array([z['raw15_en5'] for z in rr[sl]]))

# Bootstrap per source issue shared both language-view query pairs.
rng=np.random.default_rng(20261023);B=10000
issues={s:defaultdict(list) for s in ('IMVO','ISIG')}
for i in np.where(test)[0]:issues[source[i]][Q[i]['issue']].append(i)
chunks={s:[np.array(z,dtype=int) for z in issues[s].values()] for s in issues};sampled={s:rng.integers(0,len(chunks[s]),size=(B,len(chunks[s]))) for s in chunks}

def boot(diff):
 cells=[]
 for s in ('IMVO','ISIG'):
  for vv in ('en','xh'):
   nums=np.array([diff[ch][view[ch]==vv].sum() for ch in chunks[s]])
   dens=np.array([(view[ch]==vv).sum() for ch in chunks[s]])
   ind=sampled[s];cells.append(nums[ind].sum(axis=1)/np.maximum(dens[ind].sum(axis=1),1))
 vals=np.mean(cells,axis=0)
 return {'point':macro(diff,test),'ci95':list(map(float,np.quantile(vals,[.025,.975])))}
summary={};rows=[]
for cohort,(text,rt,raw,enpool) in cohorts.items():
 assert len(text)==len(rt)==len(raw)==len(enpool)==len(Q)
 pred=model.predict(v.transform(text))
 methods={'raw20':raw,'unconditional_english5':enpool,'shortLID_gate':np.where(pred==0,enpool,raw),'retrieval_evidence_gate':np.where(rt>=theta,enpool,raw)}
 sub={}
 for s,mask in [('test',test),('dev',train),('test_unique',test&unique),('test_nonfront',test&(~front))]:
  sub[s]={'n':int(mask.sum()),'macro':{name:macro(a,mask) for name,a in methods.items()},'group':{so+'_'+lg:{name:float(a[mask&(source==so)&(view==lg)].mean()) for name,a in methods.items()} for so in ('IMVO','ISIG') for lg in ('en','xh')}}
 summary[cohort]= {'summary':sub,'gain_boot':{name:boot(a.astype(float)-raw.astype(float)) for name,a in methods.items() if name!='raw20'},'LID_predict_english_rate':float(np.mean(pred==0)),'evidence_gate_english_rate':float(np.mean(rt>=theta))}
 for i,z in enumerate(Q):
  row={'cohort':cohort,'id':z['id'],'issue':z['issue'],'source':source[i],'view':view[i],'split':'test' if test[i] else 'dev','unique':bool(unique[i]),'english_by_LID':int(pred[i]==0),'english_by_evidence':int(rt[i]>=theta)}
  for name,a in methods.items():row[name]=int(a[i])
  rows.append(row)
 print(cohort,'macro',sub['test']['macro'],'LID gate rate',summary[cohort]['LID_predict_english_rate'],'evidence gate rate',summary[cohort]['evidence_gate_english_rate'],flush=True)
 print('GROUP',sub['test']['group'],flush=True)
 print('CI',summary[cohort]['gain_boot'],flush=True)
(P/'GATED_RESULTS.json').write_text(json.dumps({'protocol':open(P/'gated_protocol.sha256').read().split()[0],'frozen_clean_dev_theta':theta,'development_clean_threshold_grid':theta_dev,'cohorts':summary,'limitations':['Retrospective gate development on previously used corpus','Synthetic OCR not real OCR','Query target is archive view, not human language gold']},indent=2))
(P/'gated_results.jsonl').write_text(''.join(json.dumps(z)+'\n' for z in rows))
