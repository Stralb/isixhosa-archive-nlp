from pathlib import Path
import re,regex as rx,json,pickle,hashlib,random
from collections import defaultdict,Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from rapidfuzz.fuzz import partial_ratio

O=Path('/mnt/data/isiXhosa_phase18_evidence_routing');P=Path('/mnt/data/isiXhosa_phase17_deeper_questions')
D=pickle.load(open('/mnt/data/isiXhosa_three_track_pilot/corpus_cache.pkl','rb'));index={d['id']:i for i,d in enumerate(D)}
qen=[json.loads(x) for x in open(O/'balanced_en_queries.jsonl')];qxh=[json.loads(x) for x in open(O/'balanced_xh_queries.jsonl')]
assert len(qen)==len(qxh)==466
q=qen+qxh
for z in qen:z['view']='en'
for z in qxh:z['view']='xh'
n=len(q)
A=np.load(P/'minority_bm25_scores.npz');B=np.load(O/'balanced_xh_bm25_scores.npz')
scores={k:np.concatenate([A[k],B[k]],axis=0) for k in ('raw','en','xh')}
raw,en,xh=(scores[k] for k in ('raw','en','xh'))
assert all(z['source'] in ('IMVO','ISIG') for z in q)
# Repeat verbatim in-page paragraph gold matching, all matches count.
W=rx.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
phrase=defaultdict(list)
for i,z in enumerate(q): phrase[tuple(z['text'].lower().split())].append(i)
gold=[set() for z in q]
for j,d in enumerate(D):
 for para in re.split(r'\n\s*\n',d['raw']):
  ww=[w.lower() for w in W.findall(para)]
  for k in range(len(ww)-4):
   for i in phrase.get(tuple(ww[k:k+5]),()):gold[i].add(j)
for i,z in enumerate(q):
 assert index[z['id'].split('|')[0]] in gold[i],z['id']

# Same issue hash partition, matching within page and across newspaper episodes.
ratio=np.array([int(hashlib.sha256((z['source']+'|'+z['issue']+'|PHASE18').encode()).hexdigest()[:12],16)/float(16**12) for z in q])
train=ratio<.6;test=~train
print('groups',Counter((z['source'],z['view'],'dev' if train[i] else 'test') for i,z in enumerate(q)),flush=True)

top={k:np.argmax(v,axis=1) for k,v in scores.items()};mx={k:v[np.arange(n),top[k]] for k,v in scores.items()}
hits={k:np.array([int(top[k][i] in gold[i] and mx[k][i]>0) for i in range(n)],dtype=np.int8) for k in scores}
truth=np.array([1 if z['view']=='xh' else 0 for z in q],dtype=np.int8)
source=np.array([z['source'] for z in q]);view=np.array([z['view'] for z in q]);unique=np.array([len(g)==1 for g in gold]);front=np.array(['_p001.' in z['id'] for z in q]);

# Refit the already-specified Phase17 supervised language-id controls exactly on the same IMVO archival views. Silver labels only.
def words(s):return [w.lower() for w in W.findall(s)]
text_train_short=[];lab_short=[];text_train_long=[];lab_long=[]
for d in D:
 if d['source']!='IMVO' or not (1884<=d['year']<=1887):continue
 for lg in ('xh','en'):
  t=words(d[lg]);label=int(lg=='xh')
  if len(t)>=60:
   for st in [max(0,len(t)//4-30),max(0,3*len(t)//4-30) if len(t)>=250 else None]:
    if st is not None:text_train_long.append(' '.join(t[st:st+60]));lab_long.append(label)
  if len(t)>=10:
   rng=random.Random(int(hashlib.sha256((d['id']+lg+'shortcontrol').encode()).hexdigest()[:12],16))
   for _ in range(2):
    st=rng.randrange(len(t)-4);text_train_short.append(' '.join(t[st:st+5]));lab_short.append(label)
alltexts=[z['text'] for z in q]
models={}
for name,tx,lab,typ in [('short_logreg',text_train_short,lab_short,'tfidf'),('long_logreg',text_train_long,lab_long,'tfidf'),('long_nb',text_train_long,lab_long,'count')]:
 if typ=='tfidf':
  v=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,max_features=80000,sublinear_tf=True,dtype=np.float32)
  model=LogisticRegression(C=2.,class_weight='balanced',solver='liblinear',max_iter=300,random_state=20261023)
 else:
  v=CountVectorizer(analyzer='char',ngram_range=(3,3),min_df=2,max_features=40000,dtype=np.float32)
  model=MultinomialNB(alpha=1.)
 model.fit(v.fit_transform(tx),lab)
 pred=model.predict(v.transform(alltexts))
 if name=='short_logreg':
  prev={z['id']:int(z['pred_xho']) for z in map(json.loads,open(P/'lid_predictions_logreg_short.jsonl')) if z['kind']=='minority5'}
  assert np.array_equal(pred[:len(qen)],np.array([prev[z['id']] for z in qen]))
 models[name]=pred
 print('LID',name,'fieldlabel agreement',np.mean(pred==truth),flush=True)
 del model,v

# Identical Phase17 raw BM25 + normalized partial-ratio reranking top20; scores on xh are freshly computed, English uses saved Phase17 outcomes.
old_fuzzy={z['id']:z for z in map(json.loads,open(P/'minority_fuzzy_results.jsonl'))}
fuzzy=np.empty(n,dtype=np.int8);fuzzy5=np.empty(n,dtype=np.int8)
for i,z in enumerate(qen):
 fuzzy[i]=int(old_fuzzy[z['id']]['any1']);fuzzy5[i]=int(old_fuzzy[z['id']]['any5'])
norm=re.compile(r'[^a-z0-9]');raw_txt=[norm.sub('',d['raw'].lower()) for d in D]
for i,z in enumerate(qxh,start=len(qen)):
 cand=np.argsort(-raw[i],kind='stable')[:20];cand=[int(j) for j in cand if raw[i,j]>0]
 query=norm.sub('',z['text'].lower())
 ranks=sorted([(partial_ratio(query,raw_txt[j]),float(raw[i,j]),j) for j in cand],key=lambda t:(-t[0],-t[1],t[2]))
 order=[v[2] for v in ranks]
 fuzzy[i]=int(bool(order) and order[0] in gold[i]);fuzzy5[i]=int(any(j in gold[i] for j in order[:5]))
 if (i+1)%100==0:print('Fuzzy',i+1,flush=True)

baselines={'raw':hits['raw'],'oracle_view':np.where(truth==1,hits['xh'],hits['en']),'always_english':hits['en'],'always_xhosa':hits['xh'],'raw_fuzzy20':fuzzy}
for method,pred in models.items():baselines['LID_'+method]=np.where(pred==1,hits['xh'],hits['en'])
alpha_grid=[.125,.25,.5,.75,1,1.5,2,3,4,8,16]

def route(alpha):return np.where(alpha*mx['en']>=mx['xh'],hits['en'],hits['xh'])

def macro(a,mask):
 result=[]
 for ss in ('IMVO','ISIG'):
  for vv in ('en','xh'):
   sl=mask&(source==ss)&(view==vv)
   if sl.sum():result.append(float(a[sl].mean()))
 return float(np.mean(result))

a_scores={str(a):macro(route(a),train) for a in alpha_grid}
alpha=max(alpha_grid,key=lambda a:(a_scores[str(a)],-abs(np.log(a))))
baselines['evidence_equal']=route(1.)
baselines['evidence_calibrated']=route(alpha)
base=baselines['evidence_calibrated']
margin=np.abs(alpha*mx['en']-mx['xh'])/(alpha*mx['en']+mx['xh']+1e-9)
tau_grid=[.05,.10,.20,.30,.50]
tau_s={str(t):macro(np.where(margin<t,hits['raw'],base),train) for t in tau_grid}
tau=max(tau_grid,key=lambda t:(tau_s[str(t)],-t))
baselines['evidence_fallback']=np.where(margin<tau,hits['raw'],base)
baselines['evidence_phase18a_fixed']=route(1.5)
print('alpha dev scores',a_scores,'selected',alpha,'tau',tau,'tau scores',tau_s,flush=True)

# Same hyperparameter grid but tuned on IMVO development alone then deployed unchanged on ISIG heldout.
source_scores={str(a):macro(route(a),train&(source=='IMVO')) for a in alpha_grid}
alpha_imvo=max(alpha_grid,key=lambda a:(source_scores[str(a)],-abs(np.log(a))))
baselines['evidence_IMVO_only']=route(alpha_imvo)

rows=[]
for i,z in enumerate(q):
 row={'qid':z['id'],'issue':z['issue'],'source':source[i],'view':view[i],'split':'dev' if train[i] else 'test','source_page':z['id'].split('|')[0],'num_valid_pages':len(gold[i]),'unique':bool(unique[i]),'front':bool(front[i]),'en_max':float(mx['en'][i]),'xh_max':float(mx['xh'][i]),'raw_max':float(mx['raw'][i]),'field_chosen':'en' if alpha*mx['en'][i]>=mx['xh'][i] else 'xh','chosen_confidence':float(margin[i]),'lid_short_prediction_view':'xh' if models['short_logreg'][i] else 'en'}
 for name,v in baselines.items():row[name]=int(v[i])
 rows.append(row)
with (O/'balanced_query_results.jsonl').open('w') as f:
 for row in rows:f.write(json.dumps(row)+'\n')

# source/view primary and hard query subsets; macro over four cells.
summary={}
for partition,mask in [('dev',train),('test',test)]:
 sub={}
 for name,m in [('all',mask),('unique',mask&unique),('unique_nonfront',mask&unique&(~front)),('nonfront',mask&(~front)),('repeated',mask&(~unique))]+[(ss+'_'+vv,mask&(source==ss)&(view==vv)) for ss in ('IMVO','ISIG') for vv in ('en','xh')]:
  if m.sum():sub[name]={'n':int(m.sum()),'issues':len(set(q[i]['issue'] for i in np.where(m)[0])),'methods':{k:float(v[m].mean()) for k,v in baselines.items()}}
 sub['macro']={'n':int(mask.sum()),'methods':{k:macro(v,mask) for k,v in baselines.items()}}
 summary[partition]=sub

# Cluster bootstrap: resample issues within source (keeping all queries in issue together) and compute per-source-view mean then macro.
rng=np.random.default_rng(20261023);B=10000
boot={}
contrasts=[('evidence_calibrated','raw'),('evidence_calibrated','raw_fuzzy20'),('evidence_calibrated','LID_short_logreg'),('evidence_IMVO_only','raw_fuzzy20'),('evidence_fallback','raw_fuzzy20'),('raw_fuzzy20','raw'),('LID_short_logreg','raw')]
for area,mask in [('test',test),('test_unique',test&unique),('test_nonfront',test&(~front))]:
 groups={ss:defaultdict(list) for ss in ('IMVO','ISIG')}
 for i in np.where(mask)[0]:groups[source[i]][q[i]['issue']].append(i)
 g={ss:[np.array(inds,dtype=int) for inds in groups[ss].values()] for ss in groups}
 sampled={}
 for ss in g:
  if len(g[ss]):sampled[ss]=rng.integers(0,len(g[ss]),size=(B,len(g[ss])))
 for a,b in contrasts:
  diff=(baselines[a]-baselines[b]).astype(float)
  sums={}; counts={}
  for ss in g:
   if len(g[ss]):
    sums[ss]={vv:np.array([diff[chunk][view[chunk]==vv].sum() for chunk in g[ss]]) for vv in ('en','xh')}
    counts[ss]={vv:np.array([(view[chunk]==vv).sum() for chunk in g[ss]]) for vv in ('en','xh')}
  samples=[]
  for ss in ('IMVO','ISIG'):
   if ss not in sampled:continue
   for vv in ('en','xh'):
    nums=sums[ss][vv][sampled[ss]].sum(axis=1)
    dens=counts[ss][vv][sampled[ss]].sum(axis=1)
    samples.append((ss,vv,nums/np.maximum(dens,1)))
  arr=np.mean([s[2] for s in samples],axis=0)
  calc=macro(diff,mask)
  boot[area+'_'+a+'_minus_'+b]={'point':calc,'ci95':[float(y) for y in np.quantile(arr,[.025,.975])],'n':int(mask.sum())}

# Bootstrap per-source/view is useful for harm audit.
for ss in ('IMVO','ISIG'):
 for vv in ('en','xh'):
  m=test&(source==ss)&(view==vv)
  issues=defaultdict(list)
  for i in np.where(m)[0]:issues[q[i]['issue']].append(i)
  chunks=[np.array(t,dtype=int) for t in issues.values()]
  if not chunks:continue
  choice=rng.integers(0,len(chunks),size=(B,len(chunks)))
  for a,b in [('evidence_calibrated','raw_fuzzy20'),('evidence_calibrated','raw'),('evidence_calibrated','LID_short_logreg')]:
   d=(baselines[a]-baselines[b]).astype(float)
   sums=np.array([d[t].sum() for t in chunks]);counts=np.array([len(t) for t in chunks]);v=sums[choice].sum(axis=1)/counts[choice].sum(axis=1)
   boot['test_'+ss+'_'+vv+'_'+a+'_minus_'+b]={'point':float(d[m].mean()),'ci95':list(map(float,np.quantile(v,[.025,.975]))),'n':int(m.sum())}

out={'protocol':open(O/'balanced_protocol.sha256').read().split()[0],'n':n,'n_english':len(qen),'n_xhosa':len(qxh),'n_unique':int(unique.sum()),'n_test':int(test.sum()),'alpha_grid':alpha_grid,'dev_alpha_macro':a_scores,'alpha_selected':alpha,'dev_tau_macro':tau_s,'tau_selected':tau,'alpha_imvo_only':alpha_imvo,'imvo_dev_alpha':source_scores,'field_choice_accuracy_development':float(np.mean((alpha*mx['en'][train]>=mx['xh'][train])==(truth[train]==0))),'field_choice_accuracy_test':float(np.mean((alpha*mx['en'][test]>=mx['xh'][test])==(truth[test]==0))),'summary':summary,'bootstrap':boot,'flags':['source-view labels are not manually verified language identity','all snippets come verbatim from indexed pages','each page produces an EN and XH query','external datasets not evaluated in this phase']}
(O/'BALANCED_RESULTS.json').write_text(json.dumps(out,indent=2))
print('FINAL SUMMARY')
for area in ('dev','test'):
 for key in ('macro','all','IMVO_en','IMVO_xh','ISIG_en','ISIG_xh','unique','nonfront'):
  r=summary[area].get(key)
  if r:print(area,key,r.get('n'), {x:round(r['methods'][x]*100,1) for x in ['raw','LID_short_logreg','evidence_equal','evidence_calibrated','evidence_fallback','raw_fuzzy20','oracle_view']},flush=True)
print('BOOT')
for k,v in boot.items():
 if k.startswith('test_') and 'minus_raw_fuzzy20' in k:print(k,v,flush=True)
