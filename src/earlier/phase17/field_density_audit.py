from pathlib import Path
import pickle,json,hashlib,re,regex as rx,collections,numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
P=Path('/mnt/data/isiXhosa_phase17_deeper_questions')
D=pickle.load(open('/mnt/data/isiXhosa_three_track_pilot/corpus_cache.pkl','rb'))
W=rx.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
def words(s):return [x.lower() for x in W.findall(s)]
def paras(s):return [words(p) for p in re.split(r'\n\s*\n',s) if p.strip()]
tr=[];labels=[]
for d in D:
 if d['source']!='IMVO' or not (1884<=d['year']<=1887):continue
 for lg in ('xh','en'):
  a=words(d[lg]);
  if len(a)<60:continue
  for start in [max(0,len(a)//4-30),max(0,3*len(a)//4-30) if len(a)>=250 else None]:
   if start is not None:tr.append(' '.join(a[start:start+60]));labels.append(int(lg=='xh'))
v=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,max_features=80000,sublinear_tf=True,dtype=np.float32)
model=LogisticRegression(C=2.,class_weight='balanced',solver='liblinear',max_iter=300,random_state=20261023).fit(v.fit_transform(tr),labels)
function_words={'the','and','of','for','to','with','by','in','on','from','or','at','not','is','was','who','which','as','that','this','you','we','they','it','were','be','an','a'}
records=[]
for d in D:
 c=len(words(d['en']))
 if c<5:continue
 rp={tuple(x[j:j+5]) for x in paras(d['raw']) for j in range(len(x)-4)}
 eligible=[]
 for para in paras(d['en']):
  for i in range(len(para)-4):
   gram=tuple(para[i:i+5]);
   if gram in rp and len(set(gram))>=4 and sum(map(len,gram))>=23:eligible.append(gram)
 if not eligible:continue
 k=int(hashlib.sha256((d['id']+'densityaudit').encode()).hexdigest()[:12],16)%len(eligible)
 gram=eligible[k]
 b= '5-9' if c<10 else '10-49' if c<50 else '50-199' if c<200 else '200+'
 records.append({'page':d['id'],'source':d['source'],'issue':d['issue'],'year':d['year'],'en_word_count':c,'bucket':b,'text':' '.join(gram),'func2':int(len(set(gram)&function_words)>=2)})
print('probe sizes',len(records),flush=True)
prob=model.predict_proba(v.transform([r['text'] for r in records]))[:,1]
for r,p in zip(records,prob):r['pred_xho']=bool(p>=.5);r['field_agreement']=int(p<.5);r['p_xho']=float(p)
summary={}
for src in ('IMVO','ISIG'):
 summary[src]={}
 for b in ('5-9','10-49','50-199','200+'):
  a=[r for r in records if r['source']==src and r['bucket']==b]
  if a:
   summary[src][b]={'pages':len(a),'issues':len({r['issue'] for r in a}),'field_agreement_rate':float(np.mean([r['field_agreement'] for r in a])),'mean_prob_xho':float(np.mean([r['p_xho'] for r in a])),'func2_rate':float(np.mean([r['func2'] for r in a])), 'agreement_func2':float(np.mean([r['field_agreement'] for r in a if r['func2']])) if any(r['func2'] for r in a) else None,'agreement_nonfunc2':float(np.mean([r['field_agreement'] for r in a if not r['func2']])) if any(not r['func2'] for r in a) else None}
print(json.dumps(summary,indent=2),flush=True)
(P/'field_density_results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
(P/'field_density_summary.json').write_text(json.dumps(summary,indent=2))
