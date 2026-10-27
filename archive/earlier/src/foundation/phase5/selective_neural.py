#!/usr/bin/env python3
"""Prospectively frozen Phase 5 recurrence-type-aware weighting test.

Scientific decisions in PRECOMMIT.md. No future cohort data used in training
or model / weighting selection. Reuse the fixed Phase 4 training code.
"""
import argparse
import collections
import hashlib
import json
import math
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path('/mnt/data/isiXhosa_publication_continuation')
OUT = ROOT/'phase5'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'phase4'))
from audit_overlap import load_pages, index_ngrams, digest, RX_WORD
from pilot_charlm import doc_rep, normalize_weight_multipliers
from neural_reuse import (CONTEXT, SEEDS, EPOCHS, LR, BATCH,
                          samples_for_text, prep_evaluation, train_model,
                          evaluate, agg_domain)

TRAIN_N = 12
POLICIES = ('RAW','BALANCED','HARD','SOFT','SOFT_BALANCED','PERSIST_SOFT','TRANSIENT_SOFT')
FUTURE_RX = re.compile(r'^IMVO_(1893|1894)-(\d\d)-(\d\d)_p\d+\.txt$')


def shingle_occurrences(train):
    """Map digests to distinct issue x page-slot occurrences from TRAIN ONLY."""
    shingle_meta = collections.defaultdict(set)
    for page in train:
        match = re.search(r'_p(\d+)\.txt$',page['name'])
        assert match
        slot = int(match.group(1))
        for pos in range(max(0, len(page['tokens'])-TRAIN_N+1)):
            h = digest(page['tokens'][pos:pos+TRAIN_N])
            shingle_meta[h].add((page['issue'],slot))
    desc={}
    for h,seen in shingle_meta.items():
        issue_count = len({i for i,slot in seen})
        if issue_count < 2:
            continue
        month_count = len({i.split('_')[1][:7] for i,slot in seen})
        slot_issues = collections.defaultdict(set)
        for issue,slot in seen:
            slot_issues[slot].add(issue)
        is_persistent = issue_count>=3 and month_count>=2 and max(map(len,slot_issues.values()))>=3
        desc[h]=(issue_count,is_persistent)
    return desc


def word_class_weights(page,descs):
    toks=page['tokens']
    persist=[1]*len(toks)
    transient=[1]*len(toks)
    for pos in range(max(0,len(toks)-TRAIN_N+1)):
        hit=descs.get(digest(toks[pos:pos+TRAIN_N]))
        if hit is None: continue
        issue_count,is_persistent=hit
        output=persist if is_persistent else transient
        for j in range(pos,pos+TRAIN_N):
            if issue_count>output[j]:output[j]=issue_count
    return persist,transient


def build_training(pages, ix):
    descs=shingle_occurrences(pages)
    dlist=[]
    tallies=collections.defaultdict(float)
    for page in pages:
        d=doc_rep(page,ix)
        persist,transient=word_class_weights(page,descs)
        assert len(persist)==len(page['tokens'])
        p_weights=[];t_weights=[]
        for word,p_count,t_count in zip(page['tokens'],persist,transient):
            p_weights.extend([1/math.sqrt(p_count)]*(len(word)+1))
            t_weights.extend([1/math.sqrt(t_count)]*(len(word)+1))
            tallies[(page['source'],'persist_chars')]+=(len(word)+1)*(p_count>1)
            tallies[(page['source'],'transient_chars')]+=(len(word)+1)*(t_count>1)
        assert len(p_weights)==len(d['text'])==len(t_weights)
        d['weights']['PERSIST_SOFT']=p_weights
        d['weights']['TRANSIENT_SOFT']=t_weights
        dlist.append(d)
        tallies[(page['source'],'all_chars')]+=len(d['text'])
    # Same source balance as previous phase; normalize total effective training mass.
    fac=normalize_weight_multipliers(dlist)
    # The imported normalizer covers the five prior policies. Extend it using
    # its identical all-domain effective-mass rule for the two new policies.
    base=sum(len(d['text']) for d in dlist)
    for pol in ('PERSIST_SOFT','TRANSIENT_SOFT'):
        norm=sum(sum(d['weights'][pol]) for d in dlist)
        fac[pol]={src:base/norm for src in ('IMVO','ISIG')}
    alph={c:i for i,c in enumerate(sorted({c for d in dlist for c in d['text']} | {'?'}))}
    xs=[];ys=[];weights={p:[] for p in POLICIES}
    for d in dlist:
        x,y=samples_for_text(d['text'],alph)
        xs.append(x);ys.append(y)
        for p in POLICIES:
            w=np.asarray(d['weights'][p][CONTEXT:],dtype=np.float32)*fac[p][d['source']]
            assert len(w)==len(y)
            weights[p].append(w)
    X=np.concatenate(xs);Y=np.concatenate(ys)
    weights={p:np.concatenate(v) for p,v in weights.items()}
    assert len(X)==len(Y)==2135020,(len(X),len(Y))
    assert len(alph)==33,len(alph)
    assert len(set(map(tuple,[X[0].tolist()])))==1
    for p in POLICIES:
        assert abs(float(weights[p].mean())-1)<.003,(p,weights[p].mean())
    summary={'train_pages':len(pages),'chars':len(Y),'vocab':len(alph),
             'distinct_shingles_repeated':len(descs),
             'distinct_persistent_shingles':sum(persistent for _,persistent in descs.values()),
             'weighted_policy_means':{p:float(weights[p].mean()) for p in POLICIES},
             'matched_fraction_by_source':{source:{'persistent':tallies[(source,'persist_chars')]/tallies[(source,'all_chars')],
                             'transient':tallies[(source,'transient_chars')]/tallies[(source,'all_chars')]}
                         for source in ('IMVO','ISIG')},
             'alphabet_sha256':hashlib.sha256(''.join(alph).encode('utf8')).hexdigest()}
    return X,Y,weights,alph,summary


def load_future():
    pages=[]
    with zipfile.ZipFile('/mnt/data/TxtLangFiles1870-1894_TB.zip') as z:
        for path in z.namelist():
            if '/XhTxt/' not in path or '/._' in path or not path.endswith('.txt'):continue
            fn=path.rsplit('/',1)[-1]
            ma=FUTURE_RX.fullmatch(fn)
            if ma is None:continue
            year,month,day=ma.groups()
            pages.append({'name':fn,'source':'IMVO','issue':f'IMVO_{year}-{month}-{day}',
                          'year':int(year), 'tokens':RX_WORD.findall(z.read(path).decode('utf8','replace').casefold())})
    pages.sort(key=lambda d:d['name'])
    assert dict(collections.Counter(d['year'] for d in pages))=={1893:210,1894:190}
    return pages


def paired_bootstrap(saved,year,comparison,reference,B=3000):
    rng=np.random.default_rng(20261030+year)
    keys=sorted(saved[comparison][str(SEEDS[0])][str(year)])
    arrays={}
    for name in (comparison,reference):
        arrays[name]=np.stack([np.mean(np.stack([saved[name][str(seed)][str(year)][issue]
                        for seed in SEEDS]),axis=0) for issue in keys])
    idx=rng.integers(len(keys),size=(B,len(keys)))
    a=arrays[comparison][idx].sum(axis=1)
    b=arrays[reference][idx].sum(axis=1)
    def df(mode):
        top,den=(2,3) if mode=='nonmatch' else (0,1)
        v=a[:,top]/a[:,den]-b[:,top]/b[:,den]
        xa=arrays[comparison].sum(axis=0);xb=arrays[reference].sum(axis=0)
        per_seed=[]
        for seed in SEEDS:
            curA=np.stack([saved[comparison][str(seed)][str(year)][i] for i in keys]).sum(axis=0)
            curB=np.stack([saved[reference][str(seed)][str(year)][i] for i in keys]).sum(axis=0)
            per_seed.append(float(curA[top]/curA[den]-curB[top]/curB[den]))
        return {'delta_bpc':float(xa[top]/xa[den]-xb[top]/xb[den]),
                'ci95':np.quantile(v,[.025,.975]).tolist(),
                'paired_seed_differences':per_seed}
    return {'issue_count':len(keys),'nonmatch':df('nonmatch'),'full':df('full')}


def main(preflight=False):
    start=time.perf_counter()
    prior=load_pages()
    train=[p for p in prior if p['split']=='train']
    val=[p for p in prior if p['split']=='validation']
    assert len(train)==218 and len(val)==72
    ix,_=index_ngrams(train,TRAIN_N)
    X,Y,weights,alphabet,metadata=build_training(train,ix)
    OUT.joinpath('training_only_metadata.json').write_text(json.dumps(metadata,indent=2))
    print('TRAINING_ONLY',json.dumps(metadata),flush=True)
    validate=[dict(p,year=1887) for p in val]
    valdocs=prep_evaluation(validate,ix,alphabet)
    if preflight:
        model,loss=train_model(X,Y,weights['RAW'],'RAW',101,epochs=EPOCHS)
        summary=agg_domain(evaluate(model,valdocs),sorted({d['issue'] for d in valdocs}))
        print('PREFLIGHT_RAW_VALIDATION',json.dumps(summary),flush=True)
        OUT.joinpath('preflight_validation.json').write_text(json.dumps(summary,indent=2))
        return
    future=load_future()
    heldout_docs=prep_evaluation(future,ix,alphabet)
    issues_by_year={y:sorted({d['issue'] for d in future if d['year']==y}) for y in (1893,1894)}
    print('FUTURE_ISSUES',{y:len(i) for y,i in issues_by_year.items()},flush=True)
    results={'precommit_sha256':hashlib.sha256(OUT.joinpath('PRECOMMIT.md').read_bytes()).hexdigest(),
             'metadata':metadata,'model':'fixed prior char-MLP; 3 epochs, 3 seeds',
             'primary_contrast':'1893 Imvo nonmatch BPC PERSIST_SOFT - RAW and PERSIST_SOFT - SOFT',
             'by_policy':{},'bootstrap':{}}
    issue_cache={p:{} for p in POLICIES}
    for seed in SEEDS:
        for policy in POLICIES:
            model,loss=train_model(X,Y,weights[policy],policy,seed)
            v=evaluate(model,valdocs)
            ts=evaluate(model,heldout_docs)
            vs=agg_domain(v,sorted(v))
            tsagg={str(year):agg_domain(ts,ids) for year,ids in issues_by_year.items()}
            results['by_policy'].setdefault(policy,{})[str(seed)]={'train_bpc':loss,'validation':vs,'heldout':tsagg}
            issue_cache[policy].setdefault(str(seed),{})
            for year,ids in issues_by_year.items():
                issue_cache[policy][str(seed)][str(year)]={i:ts[i].tolist() for i in ids}
            OUT.joinpath('results_partial.json').write_text(json.dumps(results,indent=2))
            print('EVALUATED',policy,seed,tsagg,'elapsed_sec',round(time.perf_counter()-start,1),flush=True)
            del model
    for year in issues_by_year:
        results['bootstrap'][str(year)]={baseline:paired_bootstrap(issue_cache,year,'PERSIST_SOFT',baseline)
                       for baseline in POLICIES if baseline!='PERSIST_SOFT'}
    OUT.joinpath('results.json').write_text(json.dumps(results,indent=2))
    OUT.joinpath('issue_scores.json').write_text(json.dumps(issue_cache))
    print('COMPLETE',json.dumps(results['bootstrap'],indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--preflight',action='store_true')
    a=ap.parse_args()
    main(preflight=a.preflight)
