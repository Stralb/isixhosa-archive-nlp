from pathlib import Path
import pickle,json,hashlib,re,regex as rx, numpy as np
from collections import defaultdict,Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
P=Path('/mnt/data/isiXhosa_phase17_deeper_questions'); O=Path('/mnt/data/isiXhosa_phase15_source_grounded_search')
D=pickle.load(open('/mnt/data/isiXhosa_three_track_pilot/corpus_cache.pkl','rb'))
Q=[json.loads(x) for x in open(O/'paragraph_queries.jsonl')]
W=rx.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
def words(s):return [w.lower() for w in W.findall(s)]
def paraphrases(s):return [words(p) for p in re.split(r'\n\s*\n',s) if p.strip()]
# Language-evidence tag trade-off: all pages, measured by word mass, source separate.
meta=[]; summary={}
for d in D:
 rec={k:d[k] for k in ('id','issue','source','year','date','en_pct','xh_pct')}
 rec.update(en_words=len(words(d['en'])),xh_words=len(words(d['xh'])))
 meta.append(rec)
for source in ('IMVO','ISIG'):
 m=[r for r in meta if r['source']==source];summary[source]={}
 for lang in ('en','xh'):
  field=lang+'_words';mass=sum(z[field] for z in m);s=[]
  for threshold in [1,5,10,25,50,100,200]:
   eligible=[z for z in m if z[field]>=threshold]
   s.append({'threshold_words':threshold,'eligible_pages':len(eligible),'frac_pages':len(eligible)/len(m),'word_coverage':sum(z[field] for z in eligible)/mass})
  if lang=='en':
   eligible=[z for z in m if z['en_pct']>=50];s.insert(0,{'threshold_words':'English-majority','eligible_pages':len(eligible),'frac_pages':len(eligible)/len(m),'word_coverage':sum(z[field] for z in eligible)/mass})
  summary[source][lang]=dict(total_words=mass,total_pages=len(m),curve=s)
(P/'language_tag_tradeoff.json').write_text(json.dumps(summary,indent=2))
# Exactly the same early-Imvo silver-label language training as Phase 15.
train=[]; labels=[]; early_issues=set()
for d in D:
 if d['source']!='IMVO' or not (1884<=d['year']<=1887):continue
 early_issues.add(d['issue'])
 for language in ('xh','en'):
  t=words(d[language]);
  if len(t)<60:continue
  for st in [max(0,len(t)//4-30),max(0,3*len(t)//4-30) if len(t)>=250 else None]:
   if st is not None: train.append(' '.join(t[st:st+60]));labels.append(int(language=='xh'))
v=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,max_features=80000,sublinear_tf=True,dtype=np.float32)
model=LogisticRegression(C=2.,class_weight='balanced',solver='liblinear',max_iter=300,random_state=20261023).fit(v.fit_transform(train),labels)
probes=[]
for q in Q:
 probes.append({'id':q['id'],'kind':'10words','issue':q['issue'],'source':q['source'],'lang':q['lang'],'text':q['query'],'out_training_issue':q['issue'] not in early_issues})
 probes.append({'id':q['id'],'kind':'4words','issue':q['issue'],'source':q['source'],'lang':q['lang'],'text':' '.join(q['query'].split()[3:7]),'out_training_issue':q['issue'] not in early_issues})
# Previously underrepresented minority-English source pages; explicitly do not treat them as code-switched sentences.
rare=[]
for d,m in zip(D,meta):
 if not (10<=m['en_words']<50):continue
 rp={tuple(p[j:j+5]) for p in paraphrases(d['raw']) for j in range(len(p)-4)}
 candidates=[]
 for para in paraphrases(d['en']):
  for i in range(len(para)-4):
   w=tuple(para[i:i+5]);
   if len(set(w))>=4 and sum(map(len,w))>=23 and w in rp:candidates.append(w)
 if not candidates:continue
 loc=int(hashlib.sha256((d['id']+'rareenglish').encode()).hexdigest()[:8],16)%len(candidates)
 p=candidates[loc];rare.append({'id':d['id']+'|enminor','kind':'5words_English_minor','issue':d['issue'],'source':d['source'],'lang':'en','text':' '.join(p),'out_training_issue':d['issue'] not in early_issues, 'page_en_words':m['en_words']})
print('rare probes',len(rare),'by source',dict(Counter(z['source'] for z in rare)),flush=True)
probes.extend(rare)
x=v.transform([q['text'] for q in probes]);prob=model.predict_proba(x)[:,1]
for q,p in zip(probes,prob):
 q['p_xho']=float(p);q['pred_xho']=bool(p>=.5);q['correct']=int(q['pred_xho']==(q['lang']=='xh'))
s={}; kinds=sorted(set(x['kind'] for x in probes))
for kind in kinds:
 t=[x for x in probes if x['kind']==kind];s[kind]={}
 for source in ('IMVO','ISIG'):
  for lang in ('xh','en'):
   r=[x for x in t if x['source']==source and x['lang']==lang]
   if r:
    s[kind][source+'_'+lang]={'n':len(r),'correct':float(np.mean([x['correct'] for x in r])),'mean_p_xho':float(np.mean([x['p_xho'] for x in r])),'out_training_issue_n':sum(x['out_training_issue'] for x in r),'out_issue_correct':float(np.mean([x['correct'] for x in r if x['out_training_issue']])) if any(x['out_training_issue'] for x in r) else None}
(P/'language_probes_results.jsonl').write_text(''.join(json.dumps(z)+'\n' for z in probes))
(P/'language_access_summary.json').write_text(json.dumps({'language_id':s,'rare_english_counts':dict(Counter(x['source'] for x in rare)),'training_segments':len(train),'prelabel_caution':'Language-view annotations are algorithmic archive splits, not manually confirmed word-language labels'},indent=2))
print(json.dumps(s,indent=2),flush=True)
