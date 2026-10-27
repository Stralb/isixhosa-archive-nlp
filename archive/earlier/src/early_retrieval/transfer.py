"""Historical -> later-source/modern isiXhosa language-ID transfer, silver labels only."""
from pathlib import Path
import json, pickle, hashlib, regex as rx, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix
ROOT=Path('/mnt/data/isiXhosa_three_track_pilot')
with (ROOT/'corpus_cache.pkl').open('rb') as f:pages=pickle.load(f)
word=rx.compile(r'(?V1)\p{L}+(?:[\'’]\p{L}+)*')

def chunks(txt):
 t=word.findall(txt)
 if len(t)<60:return []
 output=[]
 # Source stable sampling; up to 2 widely separated 60-word windows per page
 starts=[max(0,len(t)//4-30)]
 if len(t)>=250:starts.append(max(0,3*len(t)//4-30))
 for idx in starts:output.append(' '.join(t[idx:idx+60]))
 return output
segments=[]
for r in pages:
 for lang in ('xh','en'):
  for j,text in enumerate(chunks(r[lang])):
   segments.append(dict(id=f'{r["id"]}|{lang}|{j}',page=r['id'],issue=r['issue'],source=r['source'],year=r['year'],lang=lang,text=text))
print('total segments',len(segments),flush=True)
train=[x for x in segments if x['source']=='IMVO' and 1884<=x['year']<=1887]
late=[x for x in segments if x['source']=='IMVO' and x['year']>=1888]
isig=[x for x in segments if x['source']=='ISIG' and x['year']<=1885]
isig_late=[x for x in segments if x['source']=='ISIG' and x['year']>=1887]
with open('/mnt/data/xho.jsonl') as f:modern=[json.loads(s) for s in f if s.strip()]
mod=[]
for i,row in enumerate(modern):
 for j,s in enumerate(chunks(row.get('text','') or '')):
  mod.append(dict(id=f'MODERN_{i:04d}_{j}',source='MODERN',issue=f'MODERN_{i}',year=2000,lang='xh',text=s))
print('groups', {k:(len(v),sum(x['lang']=='xh' for x in v),sum(x['lang']=='en' for x in v)) for k,v in [('train',train),('late',late),('isig',isig),('isig_late',isig_late),('modern',mod)]},flush=True)
# train only historical imvo early, labels from archive language split
v=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,max_features=80000,sublinear_tf=True,dtype=np.float32)
X=v.fit_transform([x['text'] for x in train]); y=np.array([int(x['lang']=='xh') for x in train]);
print('vector',X.shape,flush=True)
mod1=LogisticRegression(C=2.0,class_weight='balanced',solver='liblinear',random_state=20261023,max_iter=300)
mod1.fit(X,y)
# char ngram naive bayes baseline on same text, independent vectorizer
v2=CountVectorizer(analyzer='char',ngram_range=(3,3),min_df=2,max_features=40000,dtype=np.float32)
X2=v2.fit_transform([x['text'] for x in train]);mod2=MultinomialNB(alpha=1.0).fit(X2,y)
allresults=[]
for model,vec in [(mod1,v),(mod2,v2)]:
 for name,ss in [('train_replay',train),('late_IMVO',late),('early_ISIG',isig),('late_ISIG',isig_late),('modern_xho_only',mod)]:
  if not ss:continue
  Xt=vec.transform([x['text'] for x in ss]);yt=np.array([int(x['lang']=='xh') for x in ss]);prob=model.predict_proba(Xt)[:,1];yp=(prob>=.5).astype(int)
  one=dict(model='char_logreg' if model==mod1 else 'char_NB',set=name,segments=len(ss),issues=len(set(x['issue'] for x in ss)),
       true_xh=int(yt.sum()),true_en=int(len(yt)-yt.sum()),pred_xh=int(yp.sum()),
       xhosa_recall=float(np.mean(yp[yt==1]==1)) if yt.sum()>0 else None,
       english_recall=float(np.mean(yp[yt==0]==0)) if (yt==0).sum()>0 else None,
       balanced_accuracy=float(balanced_accuracy_score(yt,yp)) if len(set(yt))==2 else None,
       macro_f1=float(f1_score(yt,yp,average='macro')) if len(set(yt))==2 else None,
       mean_xho_prob=float(prob[yt==1].mean()) if yt.sum()>0 else None)
  allresults.append(one)
  print(one,flush=True)
  if name!='train_replay':
   with (ROOT/f'transfer_predictions_{one["model"]}_{name}.jsonl').open('w') as f:
    for row,p,z in zip(ss,prob,yp):f.write(json.dumps({**{k:row[k] for k in ('id','issue','source','year','lang')},'p_xh':float(p),'pred_xh':int(z)})+'\n')
(ROOT/'transfer_results.json').write_text(json.dumps(allresults,indent=2))
