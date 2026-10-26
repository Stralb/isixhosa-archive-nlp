#!/usr/bin/env python3
"""POST-HOC sensitivity checks, developed after examining primary Phase 26 outcomes.
These do not change the precommit or count as independent confirmation.
"""
import json,collections
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
import sentencepiece as spm
SEED=26024
D=Path(__file__).parent
W=[json.loads(l) for l in open(D/'windows.jsonl',encoding='utf8')]
TRAIN=[w for w in W if w['year'] in (1885,1887)]
TEST=[w for w in W if w['year']==1888]
# Publication names, recognizable subforms, identity-root terms and personal names.
# This aggressive mask may also remove genuine cultural content; sensitivity test only.
SUBSTR=('imvo','isigidi','samaxosa','zabantsundu','bantsundu','ntsundu','mnyama','xosa','jabavu','gqoba','makiwane')
EXACT={'mr','sir','lovedale','b','g','e','th'}
def mask(w):
    return [t for t in w['tokens'] if t not in EXACT and not any(part in t for part in SUBSTR)]
def clf(train, test, masked=False):
    texts=[(' '.join(mask(w)) if masked else w['text']) for w in train]
    tx=[(' '.join(mask(w)) if masked else w['text']) for w in test]
    v=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,max_features=30000,lowercase=True)
    X=v.fit_transform(texts)
    y=np.array([w['source']=='IMVO' for w in train],dtype=int)
    c=LogisticRegression(C=1,class_weight='balanced',random_state=SEED,max_iter=500,solver='liblinear').fit(X,y)
    pred=c.predict(v.transform(tx))
    gold=np.array([w['source']=='IMVO' for w in test],dtype=int)
    src_acc={s:float(np.mean(pred[[j for j,w in enumerate(test) if w['source']==s]]==gold[[j for j,w in enumerate(test) if w['source']==s]])) for s in ('IMVO','ISIG')}
    return float(balanced_accuracy_score(gold,pred)),src_acc
r=[]
for yrs,yy in [((1885,1887),1888),((1885,),1887),((1887,),1888),((1885,),1888),((1887,1888),1885)]:
    tr=[w for w in W if w['year'] in yrs];te=[w for w in W if w['year']==yy]
    for ismasked in (False,True):
        ba,sr=clf(tr,te,ismasked)
        r.append(dict(train_years=list(yrs),test_year=yy,masked=ismasked,balanced_accuracy_page=ba,source_accuracy=sr,n_test=len(te)))
print('CLASSIFIER DIAGNOSTIC',json.dumps(r,indent=2))
# Tokenizer source-specific savings after masking publication names and obvious named political identities.
rr=[]
for seed in range(3):
    ms={s:spm.SentencePieceProcessor(model_file=str(D/f'BPE_{s}_seed{seed}.model')) for s in ('IMVO','ISIG')}
    for source in ('IMVO','ISIG'):
        rows=[w for w in TEST if w['source']==source]
        chars=0;cost={s:0 for s in ms}
        for w in rows:
            txt=' '.join(mask(w));chars+=sum(ch.isalpha() for ch in txt)
            for s,model in ms.items():cost[s]+=len(model.encode(txt,out_type=int))
        a,b=100*cost['IMVO']/chars,100*cost['ISIG']/chars
        rr.append(dict(seed=seed,source=source,cost_imvo=a,cost_isig=b,correct_source_benefit=(b-a) if source=='IMVO' else (a-b),n_chars=chars))
print('TOKENIZER MASKED DIAGNOSTIC',json.dumps(rr,indent=2))
# Companion independent published-language-root measurement note only: two root strings match many surface forms.
roots=[]
for src in ('IMVO','ISIG'):
    for year in (1885,1887,1888):
        tt=[t for w in W if w['source']==src and w['year']==year for t in w['tokens']]
        for root,exact in [('ntsundu','abantsundu'),('mnyama','abamnyama')]:
            within=[t for t in tt if root in t]
            unique=sorted(collections.Counter(within).items(),key=lambda q:-q[1])[:12]
            roots.append(dict(source=src,year=year,root=root,token_instances=len(within),exact_form_instances=sum(t==exact for t in tt),distinct_surface_types=len(set(within)),top_surface_variants=unique))
(D/'posthoc_results.json').write_text(json.dumps(dict(classifier=r,tokenizer_masked=rr,published_roots_descriptive=roots),indent=2))
