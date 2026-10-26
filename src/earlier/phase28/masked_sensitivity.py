#!/usr/bin/env python3
"""Preplanned source-cue diagnostic, after the frozen Phase28 primary outcome.
Not a new confirmatory result; re-fit the Phase26 previously published cue mask.
"""
from pathlib import Path
import json,csv,collections,os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
D=Path(__file__).parent;P=Path(os.environ.get('PHASE26_DIR',str(D.parent/'isiXhosa_phase26_culture_pilots')))
W=[json.loads(x) for x in (P/'windows.jsonl').read_text(encoding='utf8').splitlines()]
SUBSTR=('imvo','isigidi','samaxosa','zabantsundu','bantsundu','ntsundu','mnyama','xosa','jabavu','gqoba','makiwane')
EXACT={'mr','sir','lovedale','b','g','e','th'}
def text(w):return ' '.join(t for t in w['tokens'] if t not in EXACT and not any(z in t for z in SUBSTR))
train=[w for w in W if w['year'] in (1885,1887)];te=[w for w in W if w['year']==1888]
vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,max_features=30000,lowercase=True)
x=vec.fit_transform([text(w) for w in train]);y=np.array([int(w['source']=='IMVO') for w in train]);m=LogisticRegression(C=1,class_weight='balanced',random_state=26024,max_iter=500,solver='liblinear').fit(x,y)
pr=m.predict(vec.transform([text(w) for w in te]));truth=np.array([int(w['source']=='IMVO') for w in te]);pred_by_file={w['file']:int(pred==truth[i]) for i,(w,pred) in enumerate(zip(te,pr))}
with open(D/'matched_1888_page_scores.csv',encoding='utf8') as f:samples=list(csv.DictReader(f))
for mode,items in [('all',te),('matched',[next(w for w in te if w['file']==r['file']) for r in samples])]:
 d=collections.defaultdict(list)
 for w in items:d[(w['source'],w['issue'])].append(pred_by_file[w['file']])
 s={s:float(np.mean([np.mean(v) for (q,_),v in d.items() if q==s])) for s in ('IMVO','ISIG')}
 print(mode,s,'issue-balanced macro=',(s['IMVO']+s['ISIG'])/2)
 (D/f'cue_mask_{mode}.json').write_text(json.dumps({'source_issue_accuracy':s,'macro':sum(s.values())/2,'n_issues':len(d),'cue_masked':True},indent=2))
with open(D/'cue_masked_page_predictions.csv','w',encoding='utf8',newline='') as f:
 wr=csv.DictWriter(f,fieldnames=['file','issue','source','correct']);wr.writeheader()
 for w in te:wr.writerow({**{k:w[k] for k in ['file','issue','source']},'correct':pred_by_file[w['file']]})
