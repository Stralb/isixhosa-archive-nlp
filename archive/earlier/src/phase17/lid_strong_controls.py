from pathlib import Path
import pickle,json,hashlib,random,regex as rx,numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
P=Path('/mnt/data/isiXhosa_phase17_deeper_questions')
D=pickle.load(open('/mnt/data/isiXhosa_three_track_pilot/corpus_cache.pkl','rb'))
M=[json.loads(x) for x in open(P/'language_probes_results.jsonl') if '5words_English_minor' in x]
A=[json.loads(x) for x in open('/mnt/data/isiXhosa_phase15_source_grounded_search/paragraph_queries.jsonl')]
W=rx.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
def words(s):return [w.lower() for w in W.findall(s)]
train_long=[];label_long=[];train_short=[];label_short=[]
for d in D:
 if d['source']!='IMVO' or not (1884<=d['year']<=1887):continue
 for lg in ('xh','en'):
  t=words(d[lg]);flag=int(lg=='xh')
  if len(t)>=60:
   for st in [max(0,len(t)//4-30),max(0,3*len(t)//4-30) if len(t)>=250 else None]:
    if st is not None:train_long.append(' '.join(t[st:st+60]));label_long.append(flag)
  if len(t)>=10:
   rng=random.Random(int(hashlib.sha256((d['id']+lg+'shortcontrol').encode()).hexdigest()[:12],16))
   for _ in range(2):
    st=rng.randrange(len(t)-4); train_short.append(' '.join(t[st:st+5]));label_short.append(flag)
print('training counts',len(train_long),len(train_short),flush=True)
q=[]
for x in A:
 q.append({'id':x['id']+'|10','source':x['source'],'lang':x['lang'],'kind':'original10','text':x['query']})
 q.append({'id':x['id']+'|4','source':x['source'],'lang':x['lang'],'kind':'original4','text':' '.join(x['query'].split()[3:7])})
for line in open(P/'language_probes_results.jsonl'):
 x=json.loads(line)
 if x['kind']=='5words_English_minor':q.append({'id':x['id'],'source':x['source'],'lang':'en','kind':'minority5','text':x['text']})
testtext=[x['text'] for x in q];truth=np.array([int(x['lang']=='xh') for x in q]);results={}
configs=[('logreg_long',train_long,label_long,'tfidf'),('nb_long',train_long,label_long,'counts'),('logreg_short',train_short,label_short,'tfidf')]
for name,t,l,type in configs:
 if type=='tfidf':
  v=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,max_features=80000,sublinear_tf=True,dtype=np.float32)
  model=LogisticRegression(C=2.,class_weight='balanced',solver='liblinear',max_iter=300,random_state=20261023)
 else:
  v=CountVectorizer(analyzer='char',ngram_range=(3,3),min_df=2,max_features=40000,dtype=np.float32)
  model=MultinomialNB(alpha=1.)
 model.fit(v.fit_transform(t),l)
 pred=model.predict(v.transform(testtext));s={}
 for kind in ('original10','original4','minority5'):
  for source in ('IMVO','ISIG'):
   for lang in ('en','xh'):
    ids=[i for i,x in enumerate(q) if x['kind']==kind and x['source']==source and x['lang']==lang]
    if ids:s[kind+'_'+source+'_'+lang]={'n':len(ids),'accuracy':float(np.mean(pred[ids]==truth[ids]))}
 results[name]=s
 with (P/('lid_predictions_'+name+'.jsonl')).open('w') as f:
  for zz,pp in zip(q,pred):f.write(json.dumps({'id':zz['id'],'kind':zz['kind'],'source':zz['source'],'lang':zz['lang'],'pred_xho':int(pp)})+'\n')
 print(name,s,flush=True)
(P/'lid_strong_controls.json').write_text(json.dumps({'results':results,'train_long_n':len(train_long),'train_short_n':len(train_short),'limitations':'Silver labels from language-separated pages; source/genre difference is not proven language error'},indent=2))
