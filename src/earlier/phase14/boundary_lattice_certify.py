#!/usr/bin/env python3
"""Exact ranking-sign bounds over disjoint local OCR character/boundary edits.

This is an exact DAG dynamic program for the stated restricted threat set,
not a general certificate for arbitrary Levenshtein edit sequences.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path
import numpy as np

P14 = Path(__file__).resolve().parent
PORTABLE = P14 / 'phase13_portable'
P13 = PORTABLE if PORTABLE.is_dir() else Path('/mnt/data/isiXhosa_phase13_ocr_certification')
sys.path.insert(0, str(P13))
from edit_distance_certificate import states_for_source, INDEX, PAIRS
if PORTABLE.is_dir():
    import sentencepiece as spm
    def models():
        return {(m, s): spm.SentencePieceProcessor(model_file=str(PORTABLE / 'models' / f'bpe_imvo_{m:03d}_seed_{s}.model'))
                for m in (0, 50, 100) for s in (11, 23, 37, 47, 59)}
else:
    from certify import models

RATES = (0., .001, .005, .01, .02)
P = len(PAIRS)
INFA = 10**8


def score_bounds_dp(original, alternatives_low, alternatives_high, joined, max_budget):
    """Return tight min/max sums for every <=K, including disjoint join edges.

    original: (N,P) contribution of unchanged word.
    alternatives_low/high: (N,P) extremal one-word character/split edits.
    joined: (N-1,P) contribution of replacing two words by their concatenation.
    """
    n,p=original.shape
    assert alternatives_low.shape==alternatives_high.shape==original.shape
    assert joined.shape==(n-1,p)
    lo=np.full((n+1,max_budget+1,p),INFA,dtype=np.int32)
    hi=np.full((n+1,max_budget+1,p),-INFA,dtype=np.int32)
    lo[0,0,:]=0;hi[0,0,:]=0
    for i in range(n):
        lo[i+1,:,:]=np.minimum(lo[i+1,:,:],lo[i,:,:]+original[i,:])
        hi[i+1,:,:]=np.maximum(hi[i+1,:,:],hi[i,:,:]+original[i,:])
        lo[i+1,1:,:]=np.minimum(lo[i+1,1:,:],lo[i,:-1,:]+alternatives_low[i,:])
        hi[i+1,1:,:]=np.maximum(hi[i+1,1:,:],hi[i,:-1,:]+alternatives_high[i,:])
        if i<n-1:
            lo[i+2,1:,:]=np.minimum(lo[i+2,1:,:],lo[i,:-1,:]+joined[i,:])
            hi[i+2,1:,:]=np.maximum(hi[i+2,1:,:],hi[i,:-1,:]+joined[i,:])
    # cumulative bounds for AT MOST k operations, rather than exactly k
    out_lo=np.minimum.accumulate(lo[n,:,:],axis=0)
    out_hi=np.maximum.accumulate(hi[n,:,:],axis=0)
    return out_lo,out_hi


def naive_relaxed_bounds(original, alternatives_low, alternatives_high, joined, max_budget):
    """Conservative but non-tight: treats overlapping local edits as independent."""
    base=original.sum(axis=0)
    single_lo=alternatives_low-original
    single_hi=alternatives_high-original
    joint_delta=joined-original[:-1]-original[1:]
    gain_lo=np.minimum(0,np.concatenate((single_lo,joint_delta),axis=0))
    gain_hi=np.maximum(0,np.concatenate((single_hi,joint_delta),axis=0))
    gain_lo=np.sort(gain_lo,axis=0)
    gain_hi=np.sort(gain_hi,axis=0)[::-1]
    out_lo=np.tile(base,(max_budget+1,1)).astype(np.int64)
    out_hi=out_lo.copy()
    for k in range(1,max_budget+1):
        out_lo[k]=out_lo[k-1]+gain_lo[k-1]
        out_hi[k]=out_hi[k-1]+gain_hi[k-1]
    return out_lo.astype(np.int32),out_hi.astype(np.int32)


def status(l,u):
    if u<0:return 'A_better'
    if l>0:return 'B_better'
    return 'abstain'


def brute_force(original, mins, maxes, joined, budget):
    n=len(original); scores=[]
    def walk(i,k,score):
        if i==n:
            scores.append(score);return
        walk(i+1,k,score+original[i])
        if k<budget:
            for z in (mins[i],maxes[i]):walk(i+1,k+1,score+z)
            if i<n-1:walk(i+2,k+1,score+joined[i])
    walk(0,0,0)
    return min(scores),max(scores)


def exhaustive_tests():
    o=np.array([[2],[-1],[3],[1]],dtype=np.int32)
    el=np.array([[-4],[-2],[-1],[-3]],dtype=np.int32)
    eh=np.array([[4],[3],[5],[2]],dtype=np.int32)
    j=np.array([[-20],[-21],[6]],dtype=np.int32)
    dl,dh=score_bounds_dp(o,el,eh,j,3)
    nl,nh=naive_relaxed_bounds(o,el,eh,j,3)
    for k in range(4):
        b=brute_force(o[:,0],el[:,0],eh[:,0],j[:,0],k)
        assert (int(dl[k,0]),int(dh[k,0]))==b,(k,dl[k],dh[k],b)
        assert nl[k,0]<=dl[k,0]<=dh[k,0]<=nh[k,0]
        if k==1:
            assert nl[k,0]==dl[k,0] and nh[k,0]==dh[k,0]
    return {str(k):{'exact':[int(dl[k,0]),int(dh[k,0])],
                    'brute':[int(q) for q in brute_force(o[:,0],el[:,0],eh[:,0],j[:,0],k)],
                    'naive':[int(nl[k,0]),int(nh[k,0])]} for k in range(4)}


def process_source(source, words, train_vocab, sp_models, previous):
    t0=time.time();n=len(words);maxk=int(max(RATES)*n)
    stats,coverage=states_for_source(words,train_vocab,sp_models)
    assert coverage['eligible_occurrences']==n
    print('SCORING boundary options',source,'n',n,'budget',maxk,flush=True)
    enc_count=0;cache={}
    def countvec(w):
        nonlocal enc_count
        if w not in cache:
            ids=[sp_models[(mix,seed)].encode(w,out_type=int) for mix in [0,50,100] for seed in [11,23,37,47,59]]
            # Excluding candidate words that become unsupported is part of specified threat set
            if any(sp_models[(mix,seed)].unk_id() in zz for (mix,seed),zz in zip(
                    [(m,s) for m in [0,50,100] for s in [11,23,37,47,59]],ids)):
                cache[w]=None
            else:cache[w]=np.array([len(zz) for zz in ids],dtype=np.int16)
            enc_count+=1
        return cache[w]
    def diff(w):
        a=countvec(w)
        if a is None:return None
        return np.asarray([(int(a[i])-int(a[j]))*(w not in train_vocab) for i,j in INDEX],dtype=np.int32)

    original=np.stack([stats[w]['observed'] for w in words]).astype(np.int32)
    char_lo=np.stack([stats[w]['min'] for w in words]).astype(np.int32)
    char_hi=np.stack([stats[w]['max'] for w in words]).astype(np.int32)
    assert np.array_equal(original.sum(axis=0), np.array([row['observed_numerator'] for row in previous['pairs']],dtype=np.int32)), 'Original numerators changed'

    # All split locations; split into two nonempty alphabetic words
    split_low={};split_high={};split_counts=0;uncovered=0
    for w in stats:
        options=[]
        for i in range(1,len(w)):
            a=diff(w[:i]);b=diff(w[i:])
            if a is not None and b is not None:options.append(a+b)
            else:uncovered+=1
        split_counts+=len(options)
        if options:
            x=np.stack(options)
            split_low[w]=x.min(axis=0)
            split_high[w]=x.max(axis=0)
        else:
            split_low[w]=stats[w]['observed'].copy()
            split_high[w]=stats[w]['observed'].copy()
    print('SPLIT variants',split_counts,'disallowed',uncovered,'elapsed',round(time.time()-t0,1),flush=True)
    single_lo=np.stack([split_low[w] for w in words]).astype(np.int32)
    single_hi=np.stack([split_high[w] for w in words]).astype(np.int32)
    joins=[];joined_unsupported=0
    for i in range(n-1):
        d=diff(words[i]+words[i+1])
        if d is None:
            joined_unsupported+=1
            # Unsupported candidate: no allowed join edge. Set join value infinity,
            # but ensure the DP never chooses it by using large finite guard.
            d=np.full(P,INFA//10,dtype=np.int32)
        joins.append(d)
    joined=np.stack(joins).astype(np.int32)
    assert joined_unsupported==0,'Need explicit disabled edges in min and max DP'
    old=previous['pairs']
    assert len(old)==P
    print('JOIN variants',n-1,'unsupported',joined_unsupported,'elapsed',round(time.time()-t0,1),flush=True)

    lo_boundary,hi_boundary=score_bounds_dp(original,single_lo,single_hi,joined,maxk)
    lo_combined,hi_combined=score_bounds_dp(original,np.minimum(single_lo,char_lo),np.maximum(single_hi,char_hi),joined,maxk)
    lo_naive,hi_naive=naive_relaxed_bounds(original,np.minimum(single_lo,char_lo),np.maximum(single_hi,char_hi),joined,maxk)
    # Character-only exactly agrees with saved Phase13 sign intervals
    def word_only_bounds(loedit,hiedit,k):
        bas=original.sum(axis=0)
        lo_gain=np.sort(np.minimum(0,loedit-original),axis=0)
        hi_gain=np.sort(np.maximum(0,hiedit-original),axis=0)[::-1]
        return bas+lo_gain[:k].sum(axis=0),bas+hi_gain[:k].sum(axis=0)
    for k in sorted(set(int(r*n) for r in RATES)):
        cl,ch=word_only_bounds(char_lo,char_hi,k)
        prior=[next(item for item in row['bounds'] if item['m']==k) for row in old]
        assert all(int(cl[p])==rr['min_numerator'] and int(ch[p])==rr['max_numerator'] for p,rr in enumerate(prior)), ('Phase13 replay mismatch',source,k)
        assert np.all(lo_naive[k]<=lo_combined[k]) and np.all(hi_naive[k]>=hi_combined[k])
        assert np.all(lo_combined[k]<=lo_boundary[k]) and np.all(hi_combined[k]>=hi_boundary[k])
        assert np.all(lo_combined[k]<=cl) and np.all(hi_combined[k]>=ch)
    assert np.array_equal(lo_boundary[0],original.sum(axis=0))
    assert np.array_equal(hi_boundary[0],original.sum(axis=0))
    assert np.all(np.diff(lo_combined,axis=0)<=0)
    assert np.all(np.diff(hi_combined,axis=0)>=0)
    seen=sum(w not in train_vocab for w in words)
    assert seen>2*maxk, 'Cannot guarantee a positive unseen denominator'
    comp=[]
    for ix,(mix_a,mix_b,seed) in enumerate(PAIRS):
        prev_pair=old[ix]
        assert (mix_a,mix_b,seed)==(prev_pair['mix_a'],prev_pair['mix_b'],prev_pair['seed'])
        rr={'mix_a':mix_a,'mix_b':mix_b,'seed':seed,'observed_numerator':int(original[:,ix].sum()),'budgets':[]}
        for rate in RATES:
            k=int(rate*n)
            c=next(z for z in prev_pair['bounds'] if z['m']==k)
            results={
              'char_only':{'lo':int(c['min_numerator']),'hi':int(c['max_numerator']),'certified':c['certified']},
              'boundary_only':{'lo':int(lo_boundary[k,ix]),'hi':int(hi_boundary[k,ix]),'certified':status(lo_boundary[k,ix],hi_boundary[k,ix])},
              'combined_dp':{'lo':int(lo_combined[k,ix]),'hi':int(hi_combined[k,ix]),'certified':status(lo_combined[k,ix],hi_combined[k,ix])},
              'naive_overlap_relaxation':{'lo':int(lo_naive[k,ix]),'hi':int(hi_naive[k,ix]),'certified':status(lo_naive[k,ix],hi_naive[k,ix])}
            }
            rr['budgets'].append({'rate':rate,'m':k,**results})
        comp.append(rr)
    sums={}
    for rate in RATES:
        rows=[next(z for z in rr['budgets'] if z['rate']==rate) for rr in comp]
        sums[str(rate)]={m:sum(z[m]['certified']!='abstain' for z in rows) for m in ['char_only','boundary_only','combined_dp','naive_overlap_relaxation']}
    return {'coverage':coverage,'observed_unseen_words':seen,'max_edits':maxk,
            'split_candidates':split_counts,'join_candidates':n-1,'new_variant_cache_size':len(cache),'runtime_s':round(time.time()-t0,2),
            'pairs':comp,'certificate_summary':sums}


def main():
    toy=exhaustive_tests();print('TOY PASS',toy,flush=True)
    f=(PORTABLE if PORTABLE.is_dir() else P13/'portable')/'frozen_inputs.json'
    frozen=json.loads(f.read_text())
    prior=json.loads((P13/'edit_distance_results.json').read_text())
    train_vocab=set(frozen['train_vocab'])
    models_loaded=models()
    result={'phase14_precommit_sha256':(P14/'PRECOMMIT.sha256').read_text().split()[0],
            'source_input_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),
            'exhaustive_toy':toy,'threat_model':'disjoint local one-character edits, word splitting, adjacent word joining; one budget unit each',
            'sources':{}}
    for src in ('ityala','zemkinkomo','modern'):
        words=frozen['sources'][src]['words']
        result['sources'][src]=process_source(src,words,train_vocab,models_loaded,prior['sources'][src])
        (P14/'results_progress.json').write_text(json.dumps(result,indent=2))
        print('COUNTS',src,result['sources'][src]['certificate_summary'],flush=True)
    (P14/'results.json').write_text(json.dumps(result,indent=2))
    (P14/'results_progress.json').unlink(missing_ok=True)
    print('FINISHED',P14/'results.json',flush=True)

if __name__=='__main__':main()
