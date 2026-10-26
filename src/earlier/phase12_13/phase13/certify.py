#!/usr/bin/env python3
"""Exact sensitivity bounds on unseen-form tokenizer-ranking differences.

Uncertainty: at most m single ASCII-letter substitutions, at most one per
word occurrence. Unseen membership is recomputed for each replacement.
"""
from __future__ import annotations
import hashlib, json, itertools, re, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import sentencepiece as spm

BASE=Path('/mnt/data/isiXhosa_phase13_ocr_certification')
OCR=Path('/mnt/data/isiXhosa_historical_eval/ocr')
MODELROOT=Path('/mnt/data/isiXhosa_historical_eval/isiXhosa_external_validation/models')
MODERN=Path('/mnt/data/xho.jsonl')
CLASSIC=Path('/mnt/data/isiXhosa_historical_eval/isiXhosa_external_validation')
sys.path.insert(0,str(CLASSIC))
from common import forms,training_pages

RATES=[0.,.001,.005,.01,.02]
SEEDS=[11,23,37,47,59]
MIXES=[0,50,100]
PAIR_MIXES=[(0,50),(0,100),(50,100)]


def best_edit_value(lam, base_n, base_s, min_u, max_u, seen_possible, size, sense, m):
    """Extremum of ratio centered F(lam), with at most m occurrence edits."""
    base=np.asarray(base_n,dtype=float)-lam*np.asarray(base_s,dtype=float)
    l=np.asarray(size,dtype=float)
    if sense=='min':
        edits=np.asarray(min_u,dtype=float)-lam*l
        edits=np.where(seen_possible,np.minimum(edits,0),edits)
        delta=np.minimum(edits-base,0)
        if m:
            # m smallest (most negative) changes; all negative values already encoded
            k=min(m,len(delta))
            gain=float(np.partition(delta,k-1)[:k].sum())
        else:gain=0.
    else:
        edits=np.asarray(max_u,dtype=float)-lam*l
        edits=np.where(seen_possible,np.maximum(edits,0),edits)
        delta=np.maximum(edits-base,0)
        if m:
            k=min(m,len(delta))
            gain=float(np.partition(delta,len(delta)-k)[-k:].sum())
        else:gain=0.
    return float(base.sum()+gain)


def exact_ratio_bounds(base_n,base_s,min_u,max_u,seen_possible,size,m,precision=1e-9):
    """Tight min/max ratio; S positive for all admissible alterations."""
    base_n=np.asarray(base_n);base_s=np.asarray(base_s);size=np.asarray(size)
    min_u=np.asarray(min_u);max_u=np.asarray(max_u)
    seen_possible=np.asarray(seen_possible,dtype=bool)
    bounds=[]
    for sense in ('min','max'):
        left,right=-5.,5.  # per-character piece-count deltas necessarily within [-5,5]
        for _ in range(43):
            mid=(left+right)*.5
            f=best_edit_value(mid,base_n,base_s,min_u,max_u,seen_possible,size,sense,m)
            if sense=='min':
                # if F(mid)>0, mid smaller than min ratio
                if f>0:left=mid
                else:right=mid
            else:
                # max F(mid)>0 iff mid < max ratio
                if f>0:left=mid
                else:right=mid
        bounds.append((left+right)*.5)
    return tuple(bounds)


def exhaustive_toy_test():
    # Each occurrence choices; (unseen numerator n, unseen denominator s)
    choices=[[(2,5),(0,0),(1,5)],[(0,0),(-1,4)],[(3,7),(1,7),(0,0)],[(1,5),(0,0),(-2,5)]]
    original=[x[0] for x in choices]
    ns=np.array([x[0] for x in original]);ss=np.array([x[1] for x in original]);
    sizes=np.array([5,4,7,5]);
    mins=np.array([min((v[0] for v in opts if v[1]>0),default=0) for opts in choices]);
    maxs=np.array([max((v[0] for v in opts if v[1]>0),default=0) for opts in choices]);
    seen=np.array([any(v[1]==0 for v in opts) for opts in choices]);
    out={}
    for m in (0,1,2):
        x,y=exact_ratio_bounds(ns,ss,mins,maxs,seen,sizes,m)
        brute=[]
        for inds in itertools.product(*[range(len(ch)) for ch in choices]):
            if sum(i!=0 for i in inds)>m:continue
            vals=[ch[i] for ch,i in zip(choices,inds)]
            s=sum(v[1] for v in vals)
            if s:brute.append(sum(v[0] for v in vals)/s)
        b0,b1=min(brute),max(brute)
        assert abs(x-b0)<1e-7 and abs(y-b1)<1e-7, (m,(x,y),(b0,b1))
        out[f'm{m}']={'analytic':[x,y],'exhaustive':[b0,b1]}
    return out


def get_sources():
    sources={}
    for key,path,skip in [
      ('ityala',OCR/'ityala_lamawele_pages.jsonl',3),
      ('zemkinkomo',OCR/'zemkinkomo_magwalandini_pages.jsonl',10)]:
        items=[]
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():continue
            row=json.loads(line)
            if int(row['pdf_page'])<=skip:continue
            ww=[w for w in forms(row['text']) if w.isascii() and w.isalpha() and len(w)>1]
            if len(ww)<60:continue
            rid=f"{key}:p{row['pdf_page']}:{row['side']}"
            items.append((rid,ww))
        selection=sorted(items,key=lambda it: hashlib.sha256(('20261023:'+it[0]).encode()).hexdigest())[:30]
        sources[key]=selection
    records=[]
    for k,line in enumerate(MODERN.read_text(encoding='utf-8').splitlines()):
        if not line.strip():continue
        row=json.loads(line)
        txt=row.get('text') or ''
        ww=[w for w in forms(txt) if w.isascii() and w.isalpha() and len(w)>1]
        if len(ww)<60:continue
        rid=f'modern:article{k:04d}'
        records.append((rid,ww))
    sources['modern']=sorted(records,key=lambda it:hashlib.sha256(('20261023:'+it[0]).encode()).hexdigest())[:30]
    return {k:{'selected_ids':[rid for rid,_ in v],
               'words':[w for rid,ww in v for w in ww[:100]],
               'total_selected_records':len(v),
               'eligible_record_count':len(v)} for k,v in sources.items()}


def models():
    return {(mix,s):spm.SentencePieceProcessor(model_file=str(MODELROOT/f'bpe_imvo_{mix:03d}_seed_{s}.model'))
            for mix in MIXES for s in SEEDS}


def build_states(words,train_words,models):
    # require original words encodable by all 15 models
    cached={}; costs={}; allowed=set()
    def costvec(w):
        if w not in cached:
            arr=[]
            for mix in MIXES:
                for s in SEEDS:
                    sp=models[(mix,s)];ids=sp.encode(w,out_type=int)
                    if sp.unk_id() in ids:
                        cached[w]=None;break
                    arr.append(len(ids))
            else:cached[w]=np.array(arr,dtype=np.int16)
        return cached[w]
    alltypes=Counter(words)
    t0=time.time()
    for w in alltypes:costvec(w)
    eligible=[w for w in words if costvec(w) is not None]
    types=Counter(eligible)
    # Allow a replacement ASCII letter only if single-letter model-encodable
    # for every model; then still validate every actual word variant.
    abc='abcdefghijklmnopqrstuvwxyz'
    replacement_alpha=[c for c in abc if costvec(c) is not None]
    states={}
    for idx,(w,freq) in enumerate(sorted(types.items())):
        variants={w}
        for i in range(len(w)):
            variants.update(w[:i]+c+w[i+1:] for c in replacement_alpha if c!=w[i])
        poss=[]
        for v in variants:
            cv=costvec(v)
            if cv is not None:
                poss.append((v,cv,v not in train_words))
        unseen_candidates=np.stack([cv for v,cv,u in poss if u]) if any(u for v,cv,u in poss) else None
        # A variant can be seen, yielding zero numerator AND zero denominator.
        seen_possible=any(not u for v,cv,u in poss)
        states[w]={'observed':costvec(w), 'unseen':w not in train_words,
                   'unseen_variants':unseen_candidates,'seen_possible':seen_possible,
                   'variant_count':len(poss),'freq':freq}
        if (idx+1)%1000==0:print('STATES',idx+1,'of',len(types),'elapsed',round(time.time()-t0,1),flush=True)
    return states,eligible,replacement_alpha,{'unique_types':len(types),'eligible_occurrences':len(eligible),
         'all_occurrences':len(words),'full_model_encodable_fraction':len(eligible)/max(1,len(words)),
         'cached_variant_types':len(cached),'elapsed_s':round(time.time()-t0,2)}


def pair_results(states,eligible,pair,seed,train_words):
    mix_a,mix_b=pair
    idx_a=MIXES.index(mix_a)*5+SEEDS.index(seed)
    idx_b=MIXES.index(mix_b)*5+SEEDS.index(seed)
    tab=[]
    for w,state in states.items():
        obs=int(state['observed'][idx_a]-state['observed'][idx_b]);
        unseen=int(state['unseen'])
        uvs=state['unseen_variants']
        if uvs is None: mn=mx=0
        else:
            diffs=uvs[:,idx_a].astype(int)-uvs[:,idx_b].astype(int)
            mn=int(diffs.min());mx=int(diffs.max())
        tab.append((w,int(state['freq']),len(w),obs,unseen,mn,mx,bool(state['seen_possible'])))
    base_n=[];base_s=[];size=[];min_u=[];max_u=[];seen_possible=[]
    for w,freq,L,obs,unseen,mn,mx,sp in tab:
        base_n.extend([obs*unseen]*freq)
        base_s.extend([L*unseen]*freq)
        size.extend([L]*freq)
        min_u.extend([mn]*freq);max_u.extend([mx]*freq)
        seen_possible.extend([sp]*freq)
    base_n=np.asarray(base_n);base_s=np.asarray(base_s);size=np.asarray(size)
    min_u=np.asarray(min_u);max_u=np.asarray(max_u);seen_possible=np.asarray(seen_possible)
    assert len(base_n)==len(eligible)
    observed=100.*base_n.sum()/base_s.sum()
    out={'source_a':mix_a,'source_b':mix_b,'seed':seed,'observed_delta_E':float(observed),
         'observed_unseen_chars':int(base_s.sum()),'N':len(eligible),'budgets':[]}
    last_low=last_high=None
    for r in RATES:
        m=int(r*len(eligible))
        # All admissible variants must keep denominator positive (worst case remove m longest unseen words)
        observed_unseen_lengths=base_s[base_s>0]
        if m:
            rem=int(np.sort(observed_unseen_lengths)[-m:].sum())
        else:rem=0
        assert base_s.sum()-rem>0,('denominator zero',pair,seed,m)
        lo,hi=exact_ratio_bounds(base_n,base_s,min_u,max_u,seen_possible,size,m)
        lo*=100.;hi*=100.
        assert lo<=(observed+1e-6) and (observed-1e-6)<=hi,(lo,observed,hi)
        if last_low is not None:
            assert lo<=last_low+1e-6 and hi>=last_high-1e-6,(lo,hi,last_low,last_high)
        last_low,last_high=lo,hi
        certified = 'A_better' if hi<0 else ('B_better' if lo>0 else 'abstain')
        out['budgets'].append({'rate':r,'m':m,'lower_E':float(lo),'upper_E':float(hi),
                                'certified_ranking':certified})
    return out


def main():
    toy=exhaustive_toy_test()
    train_vocab={w for row in training_pages() for w in forms(row['text'])}
    sources=get_sources()
    mod=models()
    result={'precommit_sha256':(BASE/'PRECOMMIT.sha256').read_text().split()[0],
            'toy_exactness':toy,
            'training_vocabulary_types':len(train_vocab),
            'sources':{},'protocol':'exact 1-char substitution per selected occurrence with <=2% bounded OCR error; historical corpora previously inspected'}
    for src,recs in sources.items():
        print('SOURCE',src, 'documents',len(recs['selected_ids']),'words',len(recs['words']),flush=True)
        state,eligible,abc,coverage=build_states(recs['words'],train_vocab,mod)
        summary={'selected_ids':recs['selected_ids'],'coverage':coverage,'replacement_alphabet':''.join(abc),
                 'eligible_occur_unseen_fraction':sum(w not in train_vocab for w in eligible)/len(eligible),
                 'pair_comparisons':[]}
        for pair in PAIR_MIXES:
            for seed in SEEDS:
                row=pair_results(state,eligible,pair,seed,train_vocab)
                summary['pair_comparisons'].append(row)
                print('RESULT',src,pair,seed,'observed',round(row['observed_delta_E'],4),
                      'cert_0.5%',row['budgets'][2]['certified_ranking'],flush=True)
        result['sources'][src]=summary
        (BASE/'results_progress.json').write_text(json.dumps(result,indent=2))
    (BASE/'results.json').write_text(json.dumps(result,indent=2))
    (BASE/'results_progress.json').unlink(missing_ok=True)
    print('COMPLETED',BASE/'results.json',flush=True)

if __name__=='__main__':main()
