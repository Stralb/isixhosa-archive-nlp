#!/usr/bin/env python3
"""Retrospective, issue-aware diagnostic: all-text vs overlap-filtered model selection.

The data, training parameters, policy definitions and selection are inherited
unchanged from pilot_charlm.py. The protocol is in SELECTION_REVERSAL_PROTOCOL.md.
"""
import collections
import hashlib
import json
import math
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from audit_overlap import load_pages, index_ngrams
from pilot_charlm import doc_rep, eval_novel_masks, normalize_weight_multipliers, build_model, ORDER

ROOT=Path('/mnt/data/isiXhosa_publication_continuation')
SEED=20261022
BS=10000
SHUFFLES=400
POLICIES=('BALANCED','HARD')
ALLOWED_SPLITS=('validation','old_test')


def compute_word_losses(model,docs,alphabet,gamma=20):
    cnts,tots=model
    V=len(alphabet)
    pseudocount=.01
    @lru_cache(maxsize=250000)
    def gram_loss(gram):
        char=gram[-1]
        p=(cnts[0][''].get(char,0.)+pseudocount)/(tots[0]['']+pseudocount*V)
        for k in range(1,ORDER+1):
            ctx=gram[ORDER-k:ORDER]
            den=tots[k].get(ctx,0.)
            if den>0:
                p=(cnts[k][ctx].get(char,0.)+gamma*p)/(den+gamma)
        return -math.log2(max(p,1e-20))
    report={}
    for d in docs:
        words=d['text'].split()
        novelmask=np.array([bool(d['eval_novel_chars'][sum(len(x)+1 for x in words[:0])])]) if False else None
        # Track word tokens and page-level true no-train-12gram mask (already assigned)
        char_pos=0
        losses=[]; counts=[]; keep=[]
        for w in words:
            wl=0.; wc=0
            wordnov=bool(d['eval_novel_chars'][char_pos]);
            for i in range(char_pos,char_pos+len(w)+1):
                if i>=ORDER:
                    wl+=gram_loss(d['text'][i-ORDER:i+1]);wc+=1
            losses.append(wl);counts.append(wc);keep.append(wordnov)
            char_pos+=len(w)+1
        assert char_pos==len(d['text'])
        report[d['name']]={'issue':d['issue'],'source':d['source'],'split':d['split'],
                           'counts':np.array(counts,dtype=np.float64),
                           'losses':np.array(losses,dtype=np.float64),
                           'keep':np.array(keep,dtype=bool)}
    return report


def by_issue(permodel,docs):
    names=set(d['name'] for d in docs)
    issue_map=collections.defaultdict(lambda:{p:np.zeros(6,dtype=np.float64) for p in POLICIES})
    for p in POLICIES:
        for name in names:
            x=permodel[p][name]
            keep=x['keep']; v=issue_map[(x['source'],x['issue'])][p]
            v[0]+=x['losses'].sum();v[1]+=x['counts'].sum()
            v[2]+=x['losses'][keep].sum();v[3]+=x['counts'][keep].sum()
            v[4]+=x['losses'][~keep].sum();v[5]+=x['counts'][~keep].sum()
    return issue_map


def get_summary(issue_map):
    out={}
    for p in POLICIES:
        bysource={}
        for src in ('IMVO','ISIG'):
            v=np.stack([val[p] for (s,_),val in issue_map.items() if s==src],axis=0).sum(axis=0)
            bysource[src]={'all':v[0]/v[1], 'nonmatch':v[2]/v[3], 'match':v[4]/v[5] if v[5]>0 else None,
                           'chars':int(v[1]),'chars_nonmatch':int(v[3]),'chars_match':int(v[5]),
                           'issues':sum(s==src for s,_ in issue_map)}
        out[p]={'domains':bysource,'macro_all':np.mean([bysource[s]['all'] for s in bysource]),
                'macro_nonmatch':np.mean([bysource[s]['nonmatch'] for s in bysource]),
                'worst_all':max(bysource[s]['all'] for s in bysource),
                'worst_nonmatch':max(bysource[s]['nonmatch'] for s in bysource)}
    return out


def bootstrap(issue_map,bs=BS,seed=SEED):
    rng=np.random.default_rng(seed);dom=[]; nissue={}
    for source in ('IMVO','ISIG'):
        vals=[item for (s,_),item in sorted(issue_map.items()) if s==source]
        A=np.stack([v['BALANCED'] for v in vals]); B=np.stack([v['HARD'] for v in vals])
        nissue[source]=len(vals)
        idx=rng.integers(0,len(vals),size=(bs,len(vals)))
        As=A[idx].sum(axis=1);Bs=B[idx].sum(axis=1)
        dom.append({'all':As[:,0]/As[:,1]-Bs[:,0]/Bs[:,1],
                    'nonmatch':As[:,2]/As[:,3]-Bs[:,2]/Bs[:,3],
                    'match':As[:,4]/np.maximum(1,As[:,5])-Bs[:,4]/np.maximum(1,Bs[:,5])})
    all_arr=(dom[0]['all']+dom[1]['all'])/2
    non_arr=(dom[0]['nonmatch']+dom[1]['nonmatch'])/2
    results={}
    for name,vals in {'macro_all':all_arr,'macro_nonmatch':non_arr,'IMVO_all':dom[0]['all'],
                      'ISIG_all':dom[1]['all'],'IMVO_nonmatch':dom[0]['nonmatch'],
                      'ISIG_nonmatch':dom[1]['nonmatch']}.items():
        q=np.quantile(vals,[.025,.5,.975]);results[name]={'ci_95':[float(q[0]),float(q[2])],
                                 'bootstrap_median':float(q[1]),'fraction_positive':float(np.mean(vals>0))}
    return {'issue_counts':nissue,'B':bs,'seed':seed,'contrasts_BALANCED_minus_HARD':results}


def sham_masks(permodel,docs,ns=SHUFFLES,seed=SEED):
    """Within-page circular shift of original no-overlap mask, keeping page topology."""
    rng=np.random.default_rng(seed)
    pageinfo={s:[] for s in ('IMVO','ISIG')}
    for d in docs:
        name=d['name']; a=permodel['BALANCED'][name];b=permodel['HARD'][name]
        assert np.array_equal(a['keep'],b['keep']) and np.array_equal(a['counts'],b['counts'])
        pageinfo[d['source']].append((a['keep'],a['counts'],a['losses']-b['losses']))
    arr=np.zeros(ns)
    for i in range(ns):
        ds=[]
        for src in ('IMVO','ISIG'):
            num=den=0.
            for keep,counts,delta in pageinfo[src]:
                shift=int(rng.integers(0,len(keep))) if len(keep)>0 else 0
                rolled=np.roll(keep,shift)
                num+=delta[rolled].sum();den+=counts[rolled].sum()
            ds.append(num/den)
        arr[i]=np.mean(ds)
    q=np.quantile(arr,[.025,.5,.95,.975])
    return {'N':ns,'seed':seed,'method':'page-wise circular-shift 12gram-nonmatch word masks',
            'delta_BPC_BALANCED_minus_HARD_quantiles':{str(p):float(v) for p,v in zip([.025,.5,.95,.975],q)},
            'sample_mean':float(arr.mean()), 'sample_min':float(arr.min()),'sample_max':float(arr.max()),
            'samples':arr}


def main():
    start=time.time()
    pages=load_pages();tr=[x for x in pages if x['split']=='train'];
    ix,_=index_ngrams(tr,12)
    docs=[doc_rep(p,ix) for p in pages];train=[d for d in docs if d['split']=='train'];
    evalu=[d for d in docs if d['split'] in ALLOWED_SPLITS]
    eval_novel_masks(evalu,ix)
    alph={c for d in train for c in d['text']};alph.add('?')
    factor=normalize_weight_multipliers(train)
    print('prepared',len(train),'training pages and',len(evalu),'evaluated pages secs',round(time.time()-start,2),flush=True)
    models={}
    for p in POLICIES:
        m=build_model(train,p,factor,alph)
        models[p]=compute_word_losses(m,evalu,alph)
        print('trained/scored',p,'secs',round(time.time()-start,2),flush=True)
        del m
    previous=json.loads((ROOT/'pilot_charlm_summary.json').read_text())
    result={'protocol':'SELECTION_REVERSAL_PROTOCOL.md',
            'status':'RETROSPECTIVE - historical 1888 outcome previously inspected',
            'training_split':'1887-01/09','validation_split':'1887-10/12','historical_test':'1888',
            'selected_all':'BALANCED','selected_no_train_12gram_match':'HARD','splits':{}}
    for split in ALLOWED_SPLITS:
        sub=[d for d in evalu if d['split']==split]
        stats=by_issue(models,sub);summary=get_summary(stats)
        # Independent implementation's recomputed total must reproduce original released pilot results.
        ref_key='validation' if split=='validation' else 'old_test'
        for p in POLICIES:
            assert abs(summary[p]['macro_all']-previous['policies'][p][ref_key]['macro_bpc'])<1e-7,(split,p,summary[p]['macro_all'])
            assert abs(summary[p]['macro_nonmatch']-previous['policies'][p][ref_key]['macro_novel_bpc'])<1e-7,(split,p,summary[p]['macro_nonmatch'])
        boot=bootstrap(stats)
        rnd=sham_masks(models,sub)
        observed=float(summary['BALANCED']['macro_nonmatch']-summary['HARD']['macro_nonmatch'])
        result['splits'][split]={'summary':summary,'issue_bootstrap':boot,
            'sham_mask':{k:v for k,v in rnd.items() if k!='samples'},
            'observed_novel_difference':observed,
            'sham_tail_fraction_ge_observed':float(np.mean(rnd['samples']>=observed)),
            'diagnostic_gate_passed':bool(split=='old_test' and observed>0 and
              boot['contrasts_BALANCED_minus_HARD']['macro_nonmatch']['ci_95'][0]>0 and
              observed>rnd['delta_BPC_BALANCED_minus_HARD_quantiles']['0.95'] and
              all(summary['BALANCED']['domains'][s]['nonmatch']>=summary['HARD']['domains'][s]['nonmatch'] for s in ('IMVO','ISIG')))}
        print(split,'observed',observed,'95% CI',boot['contrasts_BALANCED_minus_HARD']['macro_nonmatch']['ci_95'],
              'sham95',rnd['delta_BPC_BALANCED_minus_HARD_quantiles']['0.95'],
              'gate',result['splits'][split]['diagnostic_gate_passed'], 'secs',round(time.time()-start,2),flush=True)
        (ROOT/'selection_reversal_results.json').write_text(json.dumps(result,indent=2))
    print('saved',ROOT/'selection_reversal_results.json',flush=True)

if __name__=='__main__':main()
