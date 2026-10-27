#!/usr/bin/env python3
"""Controlled, retrospective character-LM weighting pilot. No new-year access."""
import collections
import json
import math
from functools import lru_cache
import time
import sys
from pathlib import Path
from audit_overlap import load_pages,index_ngrams,digest

OUT=Path('/mnt/data/isiXhosa_publication_continuation/pilot_charlm.json')
N=12
ORDER=4
GAMMAS=(5,20,100)
POLICIES=('RAW','BALANCED','HARD','SOFT','SOFT_BALANCED')


def doc_rep(p,ix):
    words=p['tokens']; counts=[1]*len(words); hard=[1.0]*len(words)
    for i in range(max(0,len(words)-N+1)):
        issues=ix.get(digest(words[i:i+N]),set())
        if len(issues)<2:continue
        issue_ids={id for id,_ in issues}
        ratio=len(issue_ids)
        keep_first=p['issue']==min(issue_ids)
        for j in range(i,i+N):
            counts[j]=max(counts[j],ratio)
            if not keep_first:hard[j]=0.
    chars=[]; weights={'RAW':[],'BALANCED':[],'HARD':[],'SOFT':[],'SOFT_BALANCED':[]}
    for word,rep,h in zip(words,counts,hard):
        tokenchars=word+' '
        chars.extend(tokenchars)
        for k in POLICIES:
            wt= h if k=='HARD' else 1/math.sqrt(rep) if k.startswith('SOFT') else 1.
            weights[k].extend([wt]*len(tokenchars))
    return {'source':p['source'],'split':p['split'],'name':p['name'],'text':''.join(chars),'weights':weights,'issue':p['issue'],'repeat_word_frac':sum(c>1 for c in counts)/max(1,len(counts))}


def normalize_weight_multipliers(train):
    # token count is fixed; rescale total effective char mass back to RAW.
    base=sum(len(d['text']) for d in train)
    vals={}
    for policy in POLICIES:
        sum_d={s:sum(sum(d['weights'][policy]) for d in train if d['source']==s) for s in ('IMVO','ISIG')}
        if policy in ('BALANCED','SOFT_BALANCED'):
            factors={s:base/(2*sum_d[s]) for s in sum_d}
        else:
            tot=sum(sum_d.values());factors={s:base/tot for s in sum_d}
        vals[policy]=factors
    return vals


def build_model(docs,policy,factors,alphabet):
    counts=[collections.defaultdict(lambda:collections.defaultdict(float)) for k in range(ORDER+1)]
    totals=[collections.defaultdict(float) for k in range(ORDER+1)]
    for d in docs:
        wtarr=d['weights'][policy];wtfactor=factors[policy][d['source']]
        seq=d['text']
        for i,ch in enumerate(seq):
            w=wtarr[i]*wtfactor
            if w==0:continue
            for order in range(ORDER+1):
                if i<order:break
                ctx=seq[i-order:i] if order>0 else ''
                counts[order][ctx][ch]+=w;totals[order][ctx]+=w
    return counts,totals


def model_evaluate(model,docs,gamma,alphabet,with_new_only=True):
    cnts,tots=model; V=len(alphabet); pseudocount=.01
    buckets={dom:[0.,0.,0.,0.] for dom in ('IMVO','ISIG')}
    @lru_cache(maxsize=250000)
    def gram_loss(gram):
        char=gram[-1]
        p=(cnts[0][''].get(char,0.)+pseudocount)/(tots[0]['']+pseudocount*V)
        for k in range(1,ORDER+1):
            ctx=gram[ORDER-k:ORDER]
            den=tots[k].get(ctx,0.)
            if den>0:p=(cnts[k][ctx].get(char,0.)+gamma*p)/(den+gamma)
        return -math.log2(max(p,1e-20))
    for d in docs:
        seq=d['text']; bucket=buckets[d['source']]
        mask=d.get('eval_novel_chars')
        for i in range(ORDER,len(seq)):
            lp=gram_loss(seq[i-ORDER:i+1]);bucket[0]+=lp;bucket[1]+=1
            if mask is not None and mask[i]:bucket[2]+=lp;bucket[3]+=1
    return {s:{'bpc_all':v[0]/v[1],'bpc_no_train_repeat':v[2]/v[3] if v[3] else None,'chars_all':v[1],'chars_novel':v[3]} for s,v in buckets.items()}


def eval_novel_masks(docs,ix):
    for d in docs:
        # map word-level exact 12-gram matches to character positions
        toks=d['text'].split(); covered=bytearray(len(toks));
        for i in range(max(0,len(toks)-N+1)):
            m=ix.get(digest(toks[i:i+N]),())
            if any(id!=d['issue'] for id,src in m):covered[i:i+N]=b'\x01'*N
        mask=[]
        for w,b in zip(toks,covered):mask.extend([not b]*(len(w)+1))
        d['eval_novel_chars']=mask
        assert len(mask)==len(d['text']),(d['name'],len(mask),len(d['text']))


def main():
    t=time.time()
    allpages=load_pages();tr=[x for x in allpages if x['split']=='train'];val=[x for x in allpages if x['split']=='validation'];tst=[x for x in allpages if x['split']=='old_test']
    ix,_=index_ngrams(tr,N)
    docs=[doc_rep(p,ix) for p in allpages]
    trd=[d for d in docs if d['split']=='train']; vald=[d for d in docs if d['split']=='validation'];tstd=[d for d in docs if d['split']=='old_test']
    eval_novel_masks(vald,ix);eval_novel_masks(tstd,ix)
    factors=normalize_weight_multipliers(trd); chars={char for d in trd for char in d['text']};chars.add('?')
    print('loaded',len(docs),'alphabet',len(chars),'secs',round(time.time()-t,1),flush=True)
    selected=list(sys.argv[1:]) or list(POLICIES)
    assert all(p in POLICIES for p in selected)
    report={'model':'interpolated 5-gram char LM; RAW validation selected gamma=20',
            'context_chars':ORDER,'training_pages':len(trd),'val_pages':len(vald),
            'old_test_pages':len(tstd),'selected_gamma':20,'weight_factors':factors,
            'policies':{},'provenance':'PILOT_SPEC.md; no independent confirmation.'}
    for p in selected:
        model=build_model(trd,p,factors,chars)
        print('trained',p,'secs',round(time.time()-t,1),flush=True)
        if p=='RAW':
            tuning=[]
            for g in GAMMAS:
                m=model_evaluate(model,vald,g,chars)
                macro=sum(m[s]['bpc_all'] for s in m)/2
                tuning.append({'gamma':g,'validation_macro_bpc':macro,'domains':m})
                print('gamma',g,'score',round(macro,5),'secs',round(time.time()-t,1),flush=True)
            gamma=min(tuning,key=lambda x:x['validation_macro_bpc'])['gamma']
            assert gamma==20
            report['gamma_tuning']=tuning
        valid=model_evaluate(model,vald,20,chars)
        test=model_evaluate(model,tstd,20,chars)
        for bucket in (valid,test):
            bucket['macro_bpc']=(bucket['IMVO']['bpc_all']+bucket['ISIG']['bpc_all'])/2
            bucket['worst_bpc']=max(bucket['IMVO']['bpc_all'],bucket['ISIG']['bpc_all'])
            bucket['macro_novel_bpc']=(bucket['IMVO']['bpc_no_train_repeat']+bucket['ISIG']['bpc_no_train_repeat'])/2
        report['policies'][p]={'validation':valid,'old_test':test}
        print('policy',p,'valid',round(valid['macro_bpc'],5),'test',round(test['macro_bpc'],5),'worst',round(test['worst_bpc'],5),'secs',round(time.time()-t,1),flush=True)
        (OUT.parent/('pilot_'+p+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2))
        del model
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2));print('wrote',OUT,flush=True)

if __name__=='__main__':main()
