#!/usr/bin/env python3
"""Prospective Phase29 word-order/shape/boilerplate ablations. No cultural labels."""
from pathlib import Path
import json, csv, collections, hashlib
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

OUT=Path(__file__).parent
SRC=Path(os.environ.get('PHASE26_DIR',str(OUT/'inputs'/'phase26')))
INP=SRC/'windows.jsonl'
W=[json.loads(x) for x in INP.read_text(encoding='utf8').splitlines()]
assert len(W)==581 and len({r['file'] for r in W})==len(W)
assert all(len(r['tokens'])==90 for r in W)
YEARS=np.array([r['year'] for r in W]); TRAIN=np.flatnonzero(np.isin(YEARS,[1885,1887])); TEST=np.flatnonzero(YEARS==1888)
S=np.array([int(r['source']=='IMVO') for r in W])
assert len(TRAIN)==374 and len(TEST)==207
EXISTING=SRC/'newspaper_test_predictions.csv'

def save_csv(name,rows):
 if not rows:return
 with (OUT/name).open('w',encoding='utf8',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)

def repeated_flags(n=13):
 owners=collections.defaultdict(set)
 for row in W:
  tt=row['tokens']
  for i in range(len(tt)-n+1):
   owners[tuple(tt[i:i+n])].add(row['issue'])
 repeated={gram for gram,issu in owners.items() if len(issu)>1}
 flags=[]
 for row in W:
  tt=row['tokens']
  flags.append(any(tuple(tt[i:i+n]) in repeated for i in range(len(tt)-n+1)))
 return np.array(flags,dtype=bool),len(repeated)

FLAG,N_REPEAT_NGRAMS=repeated_flags()
REMAIN=np.flatnonzero(~FLAG)
train_clean=np.intersect1d(TRAIN,REMAIN);test_clean=np.intersect1d(TEST,REMAIN)


def permutations(seed):
 rng=np.random.default_rng(seed)
 return [' '.join(np.array(r['tokens'])[rng.permutation(90)]) for r in W]

def shape_word(t):
 n=len(t)
 return 'a' if n<=3 else 'bb' if n<=6 else 'ccc' if n<=9 else 'dddd' if n<=12 else 'eeeee'

def shape_texts():return [' '.join(map(shape_word,r['tokens'])) for r in W]

def shape_bags():
 return np.array([[sum(shape_word(t)==k for t in r['tokens'])/90 for k in ('a','bb','ccc','dddd','eeeee')] for r in W],dtype=float)


def make_model():
 return LogisticRegression(C=1,class_weight='balanced',random_state=26024,max_iter=500,solver='liblinear')

def vec_char():
 return TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,max_features=30000,lowercase=True)

def vec_shape():
 return TfidfVectorizer(analyzer='word',ngram_range=(1,3),min_df=2,lowercase=True,token_pattern=r'(?u)\b\w+\b')


def train_pred(label,texts, train_idx=TRAIN,test_idx=TEST, feat='char',y_override=None):
 tsel=[texts[k] for k in train_idx] if feat != 'numeric' else texts[train_idx]
 esel=[texts[k] for k in test_idx] if feat != 'numeric' else texts[test_idx]
 if feat=='numeric':X,T=tsel,esel
 else:
  vec=vec_char() if feat=='char' else vec_shape()
  X=vec.fit_transform(tsel);T=vec.transform(esel)
 model=make_model(); y=S[train_idx] if y_override is None else y_override
 model.fit(X,y)
 preds=model.predict(T);pr=model.predict_proba(T)[:,1]
 rows=[]
 for j,idx in enumerate(test_idx):
  r=W[idx]
  rows.append(dict(condition=label,seed=label.split('_')[-1] if label.startswith('shuffle_') else '',file=r['file'],source=r['source'],issue=r['issue'],year=r['year'],month=r['month'],gold=int(S[idx]),pred=int(preds[j]),p_imvo=float(pr[j]),correct=int(preds[j]==S[idx]),repeated_13gram=int(FLAG[idx])))
 return rows

def aggregate(rows):
 g=collections.defaultdict(list)
 for r in rows:g[(r['source'],r['issue'])].append(r['correct'])
 acc={s:float(np.mean([np.mean(v) for (ss,i),v in g.items() if ss==s])) for s in ('IMVO','ISIG')}
 return dict(macro=(acc['IMVO']+acc['ISIG'])/2,source_issue=acc,page_bal=float(np.mean([np.mean([r['correct'] for r in rows if r['source']==src]) for src in ('IMVO','ISIG')])),n_pages=len(rows),n_issues=len(g))

def issuevals(rows):
 g=collections.defaultdict(list)
 for r in rows:g[(r['source'],r['issue'])].append(r['correct'])
 return {s:{iss:float(np.mean(v)) for (ss,iss),v in g.items() if ss==s} for s in ('IMVO','ISIG')}

def bootstrap(rows1, rows2=None, seed=29024, n=3000):
 a=issuevals(rows1);b=issuevals(rows2) if rows2 else None
 # paired bootstrap when same test issues (all conditions on same set)
 if b:
  assert {s:set(a[s]) for s in a}=={s:set(b[s]) for s in b}
 rng=np.random.default_rng(seed)
 sims=[]
 for _ in range(n):
  v=[]
  for s in ('IMVO','ISIG'):
   issues=np.array(sorted(a[s]));sample=rng.choice(issues,len(issues),replace=True)
   v.append(np.mean([a[s][z]-(b[s][z] if b else 0) for z in sample]))
  sims.append(float(np.mean(v)))
 return [float(z) for z in np.quantile(sims,[.025,.975])]

ALL=[r['text'] for r in W]
results={};every=[]
for label,texts,train,test,feat in [
 ('ordered',ALL,TRAIN,TEST,'char'),
 ('shape_ordered',shape_texts(),TRAIN,TEST,'shape'),
 ('shape_histogram',shape_bags(),TRAIN,TEST,'numeric'),
 ('no_repeat_ordered',ALL,train_clean,test_clean,'char'),
 ('ordered_retained_test',ALL,TRAIN,test_clean,'char')]:
 q=train_pred(label,texts,train,test,feat)
 every.extend(q);results[label]={**aggregate(q),'ci95':bootstrap(q)}
 print(label,results[label],flush=True)
for j,seed in enumerate([29024,29025,29026]):
 label=f'shuffle_{j}'
 q=train_pred(label,permutations(seed))
 every.extend(q);results[label]={**aggregate(q),'ci95':bootstrap(q)}
 print(label,results[label],flush=True)
 q2=train_pred(f'shuffle_retained_{j}',permutations(seed),TRAIN,test_clean)
 every.extend(q2);results[f'shuffle_retained_{j}']={**aggregate(q2),'ci95':bootstrap(q2)}

# negative control, shuffle training source labels inside year to destroy authentic source correspondence
rng=np.random.default_rng(29029);placebo=[]
vec=vec_char();Xt=vec.fit_transform([ALL[k] for k in TRAIN]);Tt=vec.transform([ALL[k] for k in TEST])
for j in range(40):
 y=S[TRAIN].copy()
 for yr in (1885,1887):
  ii=np.flatnonzero(YEARS[TRAIN]==yr);y[ii]=rng.permutation(y[ii])
 m=make_model();m.fit(Xt,y);pred=m.predict(Tt)
 r=[dict(source=W[idx]['source'],issue=W[idx]['issue'],correct=int(pred[i]==S[idx])) for i,idx in enumerate(TEST)]
 placebo.append(dict(seed=j,issue_balanced_accuracy=aggregate(r)['macro']))

# Compare held-out predictions against the prior Phase26 for frozen baseline reproduction
ref={}
with EXISTING.open('r',encoding='utf8',newline='') as f:
 for r in csv.DictReader(f):
  if r['model']=='char_wb_3_5':ref[r['file']]=(int(r['pred_imvo']),float(r['p_imvo']))
base=[x for x in every if x['condition']=='ordered']
max_abs_diff=max(abs(x['p_imvo']-ref[x['file']][1]) for x in base)
assert max_abs_diff<1e-12,max_abs_diff
assert all(x['pred']==ref[x['file']][0] for x in base)

# Paired seed-wise confidence intervals against ordered, and mean of three shuffles via per-issue scores.
base_v=issuevals(base)
shuffles=[[x for x in every if x['condition']==f'shuffle_{j}'] for j in range(3)]
shapebin=[x for x in every if x['condition']=='shape_ordered'];shapebag=[x for x in every if x['condition']=='shape_histogram']
shuffle_mean={s:{iss:float(np.mean([issuevals(rows)[s][iss] for rows in shuffles])) for iss in base_v[s]} for s in base_v}
seed_losses=[results['ordered']['macro']-results[f'shuffle_{j}']['macro'] for j in range(3)]
# Issue paired bootstrap of ordered - mean shuffled
rng=np.random.default_rng(29024)
deltas=[]
for _ in range(3000):
 q=[]
 for s in ('IMVO','ISIG'):
  issues=np.array(sorted(base_v[s]));draw=rng.choice(issues,len(issues),replace=True)
  q.append(np.mean([base_v[s][i]-shuffle_mean[s][i] for i in draw]))
 deltas.append(float(np.mean(q)))
order_delta_ci=[float(x) for x in np.quantile(deltas,[.025,.975])]
retained_base=[x for x in every if x['condition']=='ordered_retained_test']
no_rep=[x for x in every if x['condition']=='no_repeat_ordered']
summary={'data':{'total_windows':len(W),'train_windows':len(TRAIN),'test_windows':len(TEST),'train_issues':len({W[int(k)]['issue'] for k in TRAIN}),'test_issues':len({W[int(k)]['issue'] for k in TEST}),'repeated_13gram_types':N_REPEAT_NGRAMS,'repeated_windows_total':int(FLAG.sum()),'repeated_windows_by_year_source':{f'{yr}_{s}':int(sum(FLAG[k] for k,r in enumerate(W) if r['year']==yr and r['source']==s)) for yr in (1885,1887,1888) for s in ('IMVO','ISIG')},'train_retained':len(train_clean),'test_retained':len(test_clean)},'conditions':results,
'paired_ordered_minus_mean_shuffle':{'point':float(np.mean(seed_losses)),'seed_losses':list(map(float,seed_losses)),'ci95':order_delta_ci},
'paired_ordered_retained_minus_no_repeat':{'point':results['ordered_retained_test']['macro']-results['no_repeat_ordered']['macro'],'ci95':bootstrap(retained_base,no_rep)},
'null_perm':{'mean':float(np.mean([x['issue_balanced_accuracy'] for x in placebo])),'max':float(max(x['issue_balanced_accuracy'] for x in placebo)),'95pct':float(np.quantile([x['issue_balanced_accuracy'] for x in placebo],.95))},
'baseline_reproduction_max_abs_probability_diff':float(max_abs_diff),'input_sha256':hashlib.sha256(INP.read_bytes()).hexdigest()}
summary['gates']={
'H29a_word_order_dependence': all(x>=.05 for x in seed_losses) and order_delta_ci[0]>0,
'H29b_length_only_signature': any(results[z]['macro']>=.60 and results[z]['ci95'][0]>.50 for z in ('shape_ordered','shape_histogram')),
'H29c_boilerplate_excluded_signal': results['no_repeat_ordered']['macro']>=.65 and results['no_repeat_ordered']['ci95'][0]>.50,
}
(OUT/'RESULTS.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
save_csv('all_predictions.csv',every)
save_csv('null_label_permutations.csv',placebo)
print('GATES',summary['gates'],flush=True)
print('ORDER_DIFFERENCE',summary['paired_ordered_minus_mean_shuffle'],flush=True)
print('BOILERPLATE_DIFFERENCE',summary['paired_ordered_retained_minus_no_repeat'],flush=True)
