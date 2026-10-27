#!/usr/bin/env python3
"""Independent Phase27 sanity and exact-score recomputation (not a new experiment)."""
import csv,collections,hashlib,json,math
from pathlib import Path
import numpy as np
import sentencepiece as spm
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

D=Path(__file__).parent
SRC=Path('/mnt/data/isiXhosa_phase26_culture_pilots/windows.jsonl')
checks={}
W=[json.loads(x) for x in SRC.read_text(encoding='utf8').splitlines()]
assert len(W)==581
assert hashlib.sha256(SRC.read_bytes()).hexdigest()=='518868f66de6b95947c84ddd42eb180398dcea0fbb65ee8e3948b3cbae6f689f'
checks['frozen_corpus_sha256']=True
P=pd.read_csv(D/'period_predictions.csv')
A=pd.read_csv(D/'prediction_audit.csv')
checks['prediction_count']=int(len(P))
checks['fit_count']=int(len(A))
assert len(P)==17160 and len(A)==176
assert (P['month'].sub(1).mod(4)==P['fold']).all()
assert (P['year'].isin([1885,1888])).all()
assert (P['correct']==(P['prediction_year']==P['year']).astype(int)).all()
assert len(P.groupby(['file','masked','train_source','model','seed']))==17160
assert P.groupby(['file','masked','train_source','model','seed']).size().max()==1
checks['all_predictions_correctly_grouped']=True

# Verify each prediction file, date/source and issue against frozen data
Wmap={w['file']:w for w in W if w['year'] in (1885,1888)}
for row in P.itertuples():
    w=Wmap[row.file]
    assert row.issue==w['issue'] and row.month==w['month'] and row.year==w['year']
    assert row.test_source==w['source']
checks['dates_and_sources_verified']=True

# Every fold excludes the precise training evaluation month's issues.
for a in A.itertuples():
    trs=[w for w in Wmap.values() if w['source']==a.train_source and (w['month']-1)%4!=a.fold]
    tes=[w for w in Wmap.values() if (w['month']-1)%4==a.fold]
    assert len(trs)==a.n_train and len(tes)==a.n_test
    assert not ({w['issue'] for w in trs}&{w['issue'] for w in tes})
checks['month_and_issue_isolation']=True

# Full tokenizers' recorded BPE train budget, 1887-only provenance checks.
F=pd.read_csv(D/'tokenizer_fit_audit.csv')
assert len(F)==9 and (F['train_chars']==40000).all()
assert (F['year_fit']==1887).all() and (F['actual_vocab_size']==1000).all()
n_missing_training_text=0
for row in F.itertuples():
    txt=D/f'fit_1887_imvo{row.imvo_share}_seed{row.seed}.txt'
    mod=D/f'BPE_imvo{row.imvo_share}_seed{row.seed}.model'
    if txt.exists():
        assert hashlib.sha256(txt.read_bytes()).hexdigest()==row.input_sha256
        assert sum(c.isalpha() for c in txt.read_text(encoding='utf8'))==40000
    else:
        n_missing_training_text+=1  # excluded from public package; regenerate from Phase26 source
    assert hashlib.sha256(mod.read_bytes()).hexdigest()==row.model_sha256
checks['tokenizer_model_provenance_and_recorded_budget']=True
checks['training_inputs_unavailable_in_public_package']=n_missing_training_text

# Independently refit selected models and verify saved predictions/probabilities.
for kind,seed,mix,masked,fold,trsrc in [
    ('char',-1,None,0,0,'IMVO'),
    ('word',-1,None,1,3,'ISIG'),
    ('bpe',27,0,0,2,'IMVO'),
    ('bpe',29,100,1,1,'ISIG')]:
    train=[w for w in Wmap.values() if w['source']==trsrc and (w['month']-1)%4!=fold]
    test=[w for w in Wmap.values() if (w['month']-1)%4==fold]
    stop=('imvo','isigidi','samaxosa','zabantsundu','bantsundu','ntsundu','mnyama','xosa','jabavu','gqoba','makiwane')
    exclude={'mr','sir','lovedale','b','g','e','th'}
    def norm(w):
        return w['text'] if not masked else ' '.join(t for t in w['tokens'] if t not in exclude and not any(a in t for a in stop))
    ts=[norm(w) for w in train];tt=[norm(w) for w in test]
    if kind=='bpe':
        sp=spm.SentencePieceProcessor(model_file=str(D/f'BPE_imvo{mix}_seed{seed}.model'))
        ts=[' '.join(sp.encode(t,out_type=str)) for t in ts]
        tt=[' '.join(sp.encode(t,out_type=str)) for t in tt]
        v=TfidfVectorizer(lowercase=False,token_pattern=r'(?u)\S+',min_df=2,max_features=30000)
        mname=f'bpe{mix}'
    elif kind=='char':
        v=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,max_features=30000)
        mname='char'
    else:
        v=TfidfVectorizer(token_pattern=r'(?u)\b\w+\b',min_df=2,max_features=30000)
        mname='word'
    x=v.fit_transform(ts);z=v.transform(tt)
    model=LogisticRegression(C=1,solver='liblinear',class_weight='balanced',random_state=27,max_iter=500)
    model.fit(x,np.array([int(w['year']==1888) for w in train]))
    ps=model.predict_proba(z)[:,1]
    byfile=dict(zip((w['file'] for w in test),ps))
    subset=P[(P.model==mname)&(P.seed==seed)&(P.masked==masked)&(P.fold==fold)&(P.train_source==trsrc)]
    assert len(subset)==len(test)
    delta=max(abs(row.prob_1888-byfile[row.file]) for row in subset.itertuples())
    assert delta<1e-11,(kind,delta)
    checks[f'exact_refit_{kind}_{seed}_{mix}_{masked}_{fold}_{trsrc}']=delta

# Independent pandas issue-grouped recalculation (no shared summarizer code).
issue=(P.groupby(['masked','model','seed','train_source','test_source','year','issue','month'],as_index=False)
       .agg(issue_acc=('correct','mean')))
cell=(issue.groupby(['masked','model','seed','train_source','test_source','year'],as_index=False)
      .agg(mean_iss=('issue_acc','mean'))
      .groupby(['masked','model','seed','train_source','test_source'],as_index=False)
      .agg(acc=('mean_iss','mean'))
      .groupby(['masked','model','train_source','test_source'],as_index=False)
      .agg(acc=('acc','mean')))
expected=pd.read_csv(D/'cell_scores.csv')
merged=expected.merge(cell,on=['masked','model','train_source','test_source'],validate='one_to_one')
err=np.max(abs(merged['acc']-merged['issue_balanced_accuracy']))
assert len(merged)==len(expected)==40 and err<1e-12
checks['independent_pandas_recalc_max_difference']=float(err)

# Check BPE unknown tokens on each period source (not all char corpora use same coverage).
oov=[]
for seed in [27,28,29]:
    for mix in [0,50,100]:
        sp=spm.SentencePieceProcessor(model_file=str(D/f'BPE_imvo{mix}_seed{seed}.model'))
        for source in ['IMVO','ISIG']:
            rr=[w for w in Wmap.values() if w['source']==source]
            pieces=[p for w in rr for p in sp.encode(w['text'],out_type=int)]
            oov.append(dict(seed=seed,mixture=mix,source=source,
                            n_unk=sum(p==0 for p in pieces),n_pieces=len(pieces)))
assert all(x['n_pieces']>0 for x in oov)
with open(D/'bpe_unknown_tokens.csv','w',newline='',encoding='utf8') as f:
    writer=csv.DictWriter(f,fieldnames=oov[0]);writer.writeheader();writer.writerows(oov)
checks['unknown_piece_total']=int(sum(x['n_unk'] for x in oov))

(D/'VERIFICATION.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
