#!/usr/bin/env python3
"""Exact sign certificate for tokenizer unseen-form ranking under OCR edit distance <=1/word."""
import json,time,collections,hashlib
from pathlib import Path
import numpy as np
from certify import BASE,MIXES,SEEDS,PAIR_MIXES,RATES,training_pages,forms,get_sources,models


PAIRS=[(a,b,s) for a,b in PAIR_MIXES for s in SEEDS]
INDEX=[(MIXES.index(a)*5+SEEDS.index(s),MIXES.index(b)*5+SEEDS.index(s)) for a,b,s in PAIRS]


def brute_test():
    toys=[[(2,5),(0,0),(-2,5)],[(0,0),(-1,4),(1,5)],[(3,7),(0,0),(-2,6)],[(1,5),(3,6),(-1,4)]]
    original=[x[0][0] for x in toys]; minv=[min(v[0] for v in x) for x in toys];maxv=[max(v[0] for v in x) for x in toys]
    import itertools
    out={}
    for m in (0,1,2):
        changes_lo=sorted(min(0,minv[i]-original[i]) for i in range(4))[:m]
        changes_hi=sorted((max(0,maxv[i]-original[i]) for i in range(4)),reverse=True)[:m]
        lo=sum(original)+sum(changes_lo);hi=sum(original)+sum(changes_hi)
        vals=[]
        for inds in itertools.product(*[range(len(t)) for t in toys]):
            if sum(z!=0 for z in inds)>m:continue
            edit=[t[z] for t,z in zip(toys,inds)]
            if sum(s for n,s in edit)>0:vals.append(sum(n for n,s in edit))
        assert lo==min(vals) and hi==max(vals),(m,lo,hi,min(vals),max(vals))
        out[m]={'bounds':[lo,hi],'exhaustive':[min(vals),max(vals)]}
    return out


def states_for_source(words,trained,mod):
    cache={}
    def costvec(w):
        if w not in cache:
            arr=[]
            for mix in MIXES:
                for s in SEEDS:
                    sp=mod[(mix,s)];ids=sp.encode(w,out_type=int)
                    if sp.unk_id() in ids:
                        cache[w]=None;break
                    arr.append(len(ids))
            else:cache[w]=np.array(arr,dtype=np.int16)
        return cache[w]
    counts=collections.Counter(w for w in words if costvec(w) is not None)
    stats={};t0=time.monotonic()
    abc='abcdefghijklmnopqrstuvwxyz'
    for idx,(w,freq) in enumerate(sorted(counts.items())):
        vv={w}
        # All single-character substitutions
        for i in range(len(w)):
            vv.update(w[:i]+c+w[i+1:] for c in abc if c!=w[i])
            # Single-character deletion
            vv.add(w[:i]+w[i+1:])
        # Single-character insertion
        for i in range(len(w)+1):
            vv.update(w[:i]+c+w[i:] for c in abc)
        variants=[];can_unseen=[]
        for v in vv:
            if not v:continue
            cv=costvec(v)
            if cv is not None:
                variants.append(cv);can_unseen.append(v not in trained)
        x=np.stack(variants).astype(np.int16)
        novel=np.asarray(can_unseen,dtype=bool)
        orig=costvec(w)
        obs=np.array([(orig[i]-orig[j])*(w not in trained) for i,j in INDEX],dtype=int)
        # Possible unseen-group numerical contribution across substitutions, insertions, deletions; 0 for seen
        pairvals=(x[:,[i for i,j in INDEX]]-x[:,[j for i,j in INDEX]]).astype(np.int16)
        pairvals[~novel,:]=0
        mn=pairvals.min(axis=0).astype(int);mx=pairvals.max(axis=0).astype(int)
        stats[w]={'n':freq,'observed':obs,'min':mn,'max':mx,'variants':len(variants)}
        if (idx+1)%600==0:print('EDITS STATES',idx+1,'/',len(counts),'cache',len(cache),'seconds',round(time.monotonic()-t0,1),flush=True)
    return stats,{'eligible_occurrences':sum(counts.values()),'all_occurrences':len(words),'unique_types':len(counts),'variant_cache_size':len(cache),'elapsed_s':round(time.monotonic()-t0,1)}


def main():
    toy=brute_test()
    prev=json.loads((BASE/'results.json').read_text())
    trained={w for row in training_pages() for w in forms(row['text'])}
    docs=get_sources();mod=models()
    res={'extension_precommit_sha256':(BASE/'EDIT_DISTANCE_EXTENSION.sha256').read_text().split()[0],
         'toy':toy,'sources':{},'comparison':'addition of single-character insertions, deletions and substitutions, compared with substitution-only exact bounds'}
    for src,rows in docs.items():
        print('SOURCE',src,'word_occurrences',len(rows['words']),flush=True)
        states,coverage=states_for_source(rows['words'],trained,mod)
        ext=[]
        base_arr=[[] for _ in PAIRS];lower_arr=[[] for _ in PAIRS];upper_arr=[[] for _ in PAIRS]
        N=coverage['eligible_occurrences']
        for s in states.values():
            for ix in range(len(PAIRS)):
                base_arr[ix].extend([int(s['observed'][ix])]*s['n'])
                lower_arr[ix].extend([int(s['min'][ix])]*s['n'])
                upper_arr[ix].extend([int(s['max'][ix])]*s['n'])
        assert all(len(x)==N for x in base_arr)
        old=prev['sources'][src]['pair_comparisons']
        for ix,(a,b,seed) in enumerate(PAIRS):
            obs=np.array(base_arr[ix]); low=np.array(lower_arr[ix]);high=np.array(upper_arr[ix])
            lo_gain=np.sort(np.minimum(0,low-obs));hi_gain=np.sort(np.maximum(0,high-obs))[::-1]
            oc=int(obs.sum());old_row=next(rr for rr in old if rr['seed']==seed and rr['source_a']==a and rr['source_b']==b)
            result={'mix_a':a,'mix_b':b,'seed':seed,'observed_numerator':oc,'bounds':[]}
            prevcert=None
            for rate in RATES:
                m=int(rate*N)
                minnum=int(oc+lo_gain[:m].sum());maxnum=int(oc+hi_gain[:m].sum())
                assert minnum<=oc<=maxnum
                if prevcert is not None:
                    assert minnum<=prevcert[0] and maxnum>=prevcert[1]
                prevcert=(minnum,maxnum)
                cert='A_better' if maxnum<0 else ('B_better' if minnum>0 else 'abstain')
                oldcert=next(br['certified_ranking'] for br in old_row['budgets'] if br['rate']==rate)
                # enlarged uncertainty set cannot turn a non-certification into certification
                if oldcert=='abstain':assert cert=='abstain',(src,rate,a,b,seed,cert,oldcert)
                if rate==0 and oc==0: assert cert=='abstain'
                result['bounds'].append({'rate':rate,'m':m,'min_numerator':minnum,'max_numerator':maxnum,'certified':cert})
            ext.append(result)
            print('RESULT',src,(a,b),seed,'observed_numerator',oc,'edit-0.5%',result['bounds'][2]['certified'],flush=True)
        res['sources'][src]={'coverage':coverage,'pairs':ext}
        (BASE/'edit_distance_progress.json').write_text(json.dumps(res,indent=2))
    (BASE/'edit_distance_results.json').write_text(json.dumps(res,indent=2))
    (BASE/'edit_distance_progress.json').unlink(missing_ok=True)
    print('FINISHED',BASE/'edit_distance_results.json',flush=True)

if __name__=='__main__':main()
