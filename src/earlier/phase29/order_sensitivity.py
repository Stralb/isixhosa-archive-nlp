#!/usr/bin/env python3
"""Predeclared Phase29B local-word-order test (does not validate cultural discourse)."""
from pathlib import Path
import collections,csv,json,os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
D=Path(__file__).parent; P=Path(os.environ.get('PHASE26_DIR',str(D/'inputs'/'phase26'))); W=[json.loads(x) for x in (P/'windows.jsonl').read_text().splitlines()]
train=[i for i,r in enumerate(W) if r['year'] in (1885,1887)]; test=[i for i,r in enumerate(W) if r['year']==1888]
y=np.array([int(r['source']=='IMVO') for r in W]); ordered=[r['text'] for r in W]

def perm(seed):
 rng=np.random.default_rng(seed)
 return [' '.join(np.array(r['tokens'])[rng.permutation(90)]) for r in W]

def vectorizer(k):
 kwargs={'min_df':2,'max_features':30000,'lowercase':True}
 if k=='word1':return TfidfVectorizer(analyzer='word',ngram_range=(1,1),token_pattern=r'(?u)\b[^\W\d_]+\b',**kwargs)
 if k=='word12':return TfidfVectorizer(analyzer='word',ngram_range=(1,2),token_pattern=r'(?u)\b[^\W\d_]+\b',**kwargs)
 return TfidfVectorizer(analyzer='char',ngram_range=(3,5),**kwargs)

def fit(k,mode,docs):
 v=vectorizer(k)
 X=v.fit_transform([docs[i] for i in train]);T=v.transform([docs[i] for i in test]);model=LogisticRegression(C=1,class_weight='balanced',solver='liblinear',max_iter=500,random_state=26024)
 model.fit(X,y[train]);p=model.predict_proba(T)[:,1]
 rows=[]
 for idx,pr in zip(test,p):
  r=W[idx]; rows.append(dict(condition=k,mode=mode,file=r['file'],source=r['source'],issue=r['issue'],gold=int(y[idx]),pred=int(pr>=.5),p_imvo=float(pr),correct=int(int(pr>=.5)==y[idx])))
 return rows

def means(rows):
 g=collections.defaultdict(list)
 for r in rows:g[(r['source'],r['issue'])].append(r['correct'])
 per={s:np.mean([np.mean(v) for (ss,_),v in g.items() if ss==s]) for s in ('IMVO','ISIG')}
 return (float(np.mean(list(per.values()))),{k:float(v) for k,v in per.items()})

def issue_values(rows):
 g=collections.defaultdict(list)
 for r in rows:g[(r['source'],r['issue'])].append(r['correct'])
 return {ss:{issue:float(np.mean(v)) for (s,issue),v in g.items() if s==ss} for ss in ('IMVO','ISIG')}

def ci_diff(rowsA, rowsB_list,seed=29224):
 A=issue_values(rowsA);B=[issue_values(z) for z in rowsB_list]
 rng=np.random.default_rng(seed);vals=[]
 for _ in range(3000):
  q=[]
  for s in ('IMVO','ISIG'):
   iss=np.array(sorted(A[s]));sample=rng.choice(iss,len(iss),replace=True)
   q.append(float(np.mean([A[s][i]-np.mean([b[s][i] for b in B]) for i in sample])))
  vals.append(np.mean(q))
 return [float(q) for q in np.quantile(vals,[.025,.975])]

conditions={}; allrows=[]
for k in ('word1','word12','cross_char'):
 for mode,docs in [('ordered',ordered)]+[(f'shuffle_{j}',perm(seed)) for j,seed in enumerate((29024,29025,29026))]:
  rows=fit(k,mode,docs); conditions[(k,mode)]=rows;allrows.extend(rows)
  print(k,mode,means(rows)[0],flush=True)

# Word-unigram exact invariant and Phase26 reproduction.
ref={}
with (P/'newspaper_test_predictions.csv').open(newline='') as f:
 for r in csv.DictReader(f):
  if r['model']=='word_unigram':ref[r['file']]=float(r['p_imvo'])
orig=conditions[('word1','ordered')]
maxabs=max(abs(r['p_imvo']-ref[r['file']]) for r in orig)
assert maxabs<1e-12,maxabs
maxshuffle=max(abs(a['p_imvo']-b['p_imvo']) for j in range(3) for a,b in zip(orig,conditions[('word1',f'shuffle_{j}')]))
assert maxshuffle<1e-12,maxshuffle

summary={'data':{'train':len(train),'test':len(test),'test_issues':len({W[i]['issue'] for i in test})},'ordered':{},'shuffled':{},'paired_comparisons':{},'word_unigram_max_abs_replay_diff':maxabs,'word_unigram_max_abs_shuffle_diff':maxshuffle}
for k in ('word1','word12','cross_char'):
 summary['ordered'][k]=means(conditions[(k,'ordered')])[0]
 summary['shuffled'][k]=[means(conditions[(k,f'shuffle_{j}')])[0] for j in range(3)]
 t=conditions[(k,'ordered')];s=[conditions[(k,f'shuffle_{j}')] for j in range(3)]
 summary['paired_comparisons'][k+'_ordered_minus_shuffled']={'point':float(means(t)[0]-np.mean([means(x)[0] for x in s])),'ci95':ci_diff(t,s)}
summary['paired_comparisons']['word12_ordered_minus_word1_ordered']={'point':summary['ordered']['word12']-summary['ordered']['word1'],'ci95':ci_diff(conditions[('word12','ordered')],[conditions[('word1','ordered')]])}
p=summary['paired_comparisons']
summary['gates']={'H29B1_bigram_beats_unigram':p['word12_ordered_minus_word1_ordered']['point']>=.05 and p['word12_ordered_minus_word1_ordered']['ci95'][0]>0,
'H29B2_bigram_order_effect':p['word12_ordered_minus_shuffled']['point']>=.05 and p['word12_ordered_minus_shuffled']['ci95'][0]>0,
'H29B3_cross_char_order_effect':p['cross_char_ordered_minus_shuffled']['point']>=.05 and p['cross_char_ordered_minus_shuffled']['ci95'][0]>0}
(D/'ORDER_SENSITIVITY_RESULTS.json').write_text(json.dumps(summary,indent=2))
with (D/'order_sensitivity_predictions.csv').open('w',encoding='utf8',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
print('RESULTS',json.dumps(summary['paired_comparisons'],indent=2));print('GATES',summary['gates'])
