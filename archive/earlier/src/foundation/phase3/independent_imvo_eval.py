#!/usr/bin/env python3
"""Independent 1889 primary + 1890 secondary Imvo-only temporal test.

1887 charLM specification and fixed validation selections are unchanged.
This code must not use 1888 data to fit, calibrate or choose models.
See PHASE3_PRECOMMIT.md for the prospective design.
"""
from __future__ import annotations
import collections
import json
import math
import re
import time
import zipfile
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit_overlap import load_pages, index_ngrams, digest, RX_WORD
from pilot_charlm import (doc_rep, normalize_weight_multipliers, build_model,
                           model_evaluate, eval_novel_masks, ORDER, POLICIES)

ROOT=Path('/mnt/data/isiXhosa_publication_continuation')
OUT=ROOT/'phase3'
ZIP=Path('/mnt/data/TxtLangFiles1870-1894_TB.zip')
SEED=20261023
B=5000
N=12
YEARS=(1889,1890)
TARGET_RX=re.compile(r'^IMVO_(1889|1890)-(\d\d)-(\d\d)_p\d+\.txt$')


def load_future_pages():
    pages=[]
    with zipfile.ZipFile(ZIP) as z:
        for path in z.namelist():
            if '/XhTxt/' not in path or '/._' in path or not path.endswith('.txt'):
                continue
            filename=path.rsplit('/',1)[-1]
            m=TARGET_RX.fullmatch(filename)
            if m is None:continue
            yr,mo,da=m.groups()
            words=RX_WORD.findall(z.read(path).decode('utf-8','replace').casefold())
            pages.append({'name':filename,'source':'IMVO','split':'independent_'+yr,
                          'issue':'IMVO_'+yr+'-'+mo+'-'+da,
                          'date':yr+'-'+mo+'-'+da,'tokens':words})
    pages.sort(key=lambda x:x['name'])
    counts=collections.Counter(p['date'][:4] for p in pages)
    assert dict(counts)=={'1889':199,'1890':220},counts
    return pages


def prep_future(pages,ngram_index,train_chars):
    docs=[]
    for p in pages:
        word=p['tokens']
        covered=bytearray(len(word))
        for i in range(max(0,len(word)-N+1)):
            ss=ngram_index.get(digest(word[i:i+N]),())
            if any(issue!=p['issue'] for issue,source in ss):
                covered[i:i+N]=b'\x01'*N
        text=' '.join(word)+' ' if word else ''
        mask=bytearray(len(text))
        pos=0
        for w,b in zip(word,covered):
            span=len(w)+1
            if not b: mask[pos:pos+span]=b'\x01'*span
            pos+=span
        assert pos==len(text)
        docs.append({'text':text,'mask':mask,'issue':p['issue'], 'name':p['name'],
                     'year':int(p['date'][:4]),'oov_chars':sum(x not in train_chars for x in text),
                     'num_words':len(word),'repeat_words':sum(covered)})
    return docs


def score(model,docs,alphabet,policy,selected_gamma=20):
    cnts,tots=model
    V=len(alphabet)
    pseudocount=.01
    @lru_cache(maxsize=250000)
    def gram_loss(gram):
        ch=gram[-1]
        prob=(cnts[0][''].get(ch,0.)+pseudocount)/(tots[0]['']+pseudocount*V)
        for order in range(1,ORDER+1):
            ctx=gram[ORDER-order:ORDER]
            denom=tots[order].get(ctx,0.)
            if denom>0:prob=(cnts[order][ctx].get(ch,0.)+selected_gamma*prob)/(denom+selected_gamma)
        return -math.log2(max(prob,1e-20))
    by_issue=collections.defaultdict(lambda:np.zeros(6,dtype=np.float64))
    for count,d in enumerate(docs,1):
        s=d['text'];mask=d['mask']
        v=by_issue[(d['year'],d['issue'])]
        for i in range(ORDER,len(s)):
            loss=gram_loss(s[i-ORDER:i+1]);v[0]+=loss;v[1]+=1
            if mask[i]:v[2]+=loss;v[3]+=1
            else:v[4]+=loss;v[5]+=1
    return by_issue


def val_reproduction(model,valdocs,alphabet,policy,prior):
    metrics=model_evaluate(model,valdocs,20,alphabet)
    expected=prior['policies'][policy]['validation']
    for domain in ('IMVO','ISIG'):
        assert abs(metrics[domain]['bpc_all']-expected[domain]['bpc_all'])<1e-8, (domain,policy,metrics[domain],expected[domain])
        assert abs(metrics[domain]['bpc_no_train_repeat']-expected[domain]['bpc_no_train_repeat'])<1e-8
    return {'IMVO':metrics['IMVO']['bpc_all'], 'ISIG':metrics['ISIG']['bpc_all']}


def aggregate_year(issues,year,policy):
    vals=np.stack([v for (p,yr,iss),v in issues.items() if p==policy and yr==year],axis=0)
    total=vals.sum(axis=0)
    return {'all':total[0]/total[1], 'nonmatch':total[2]/total[3],
            'match':total[4]/total[5] if total[5]>0 else None,
            'char_all':int(total[1]), 'char_nonmatch':int(total[3]),
            'char_match':int(total[5]), 'matched_fraction':total[5]/total[1],
            'issue_count':len(vals)}


def bootstrap(issues,year,selection,contrast_a='BALANCED',contrast_b='HARD'):
    rng=np.random.default_rng(SEED+year)
    rows=sorted({iss for (p,yr,iss) in issues if yr==year})
    a=np.array([issues[(contrast_a,year,iss)] for iss in rows]);b=np.array([issues[(contrast_b,year,iss)] for iss in rows]);n=len(rows)
    idx=rng.integers(n,size=(B,n))
    suma=a[idx].sum(axis=1); sumb=b[idx].sum(axis=1)
    result={}
    for measure,(num,den) in {'all':(0,1),'nonmatch':(2,3),'match':(4,5)}.items():
        contrasts=suma[:,num]/np.maximum(suma[:,den],1)-sumb[:,num]/np.maximum(sumb[:,den],1)
        result[measure]={'ci_95':np.quantile(contrasts,[.025,.975]).tolist(),
                         'median':float(np.median(contrasts)),
                         'fraction_positive':float(np.mean(contrasts>0))}
    # Also, fixed selector policy versus conventional baseline
    for a_label,b_label in [('robust','full_macro'),('robust','nonmatch_macro')]:
        a_policy=selection[a_label];b_policy=selection[b_label]
        av=np.array([issues[(a_policy,year,iss)] for iss in rows]);bv=np.array([issues[(b_policy,year,iss)] for iss in rows]);
        an=av[idx].sum(axis=1);bn=bv[idx].sum(axis=1)
        d=bn[:,2]/bn[:,3] - an[:,2]/an[:,3] # positive means robust superior
        result[a_label+'_advantage_over_'+b_label]={'policies':[a_policy,b_policy],
            'ci_95':np.quantile(d,[.025,.975]).tolist(),
            'point':float((bv.sum(axis=0)[2]/bv.sum(axis=0)[3])-(av.sum(axis=0)[2]/av.sum(axis=0)[3])),
            'fraction_positive':float(np.mean(d>0))}
    return result


def main():
    start=time.time()
    prior=json.loads((ROOT/'pilot_charlm_summary.json').read_text())
    selection=json.loads((OUT/'selector_preflight.json').read_text())['selected']
    # Before opening future year content, build and validate from old 1887 data only.
    old=load_pages()
    train=[x for x in old if x['split']=='train']
    val=[x for x in old if x['split']=='validation']
    assert len(train)==218 and len(val)==72
    idx,_=index_ngrams(train,N)
    train_docs=[doc_rep(p,idx) for p in train]
    val_docs=[doc_rep(p,idx) for p in val]
    eval_novel_masks(val_docs,idx)
    factors=normalize_weight_multipliers(train_docs)
    alphabet=set(''.join(x['text'] for x in train_docs))|{'?'}
    for policy in POLICIES:
        for source in ('IMVO','ISIG'):
            assert abs(factors[policy][source]-prior['weight_factors'][policy][source])<1e-12
    print('train',len(train),'val',len(val),'alphabet',len(alphabet),'sec',round(time.time()-start,1),flush=True)
    # Future-year access is now allowed. Still no fitting/tuning on independent data.
    future=load_future_pages()
    evaldocs=prep_future(future,idx,alphabet)
    audit={}
    for year in YEARS:
        subset=[d for d in evaldocs if d['year']==year]
        audit[str(year)]={'pages':len(subset),'issues':len({d['issue'] for d in subset}),
               'surface_words':sum(d['num_words'] for d in subset),
               'repeat_words':sum(d['repeat_words'] for d in subset),
               'out_of_train_alphabet_chars':sum(d['oov_chars'] for d in subset)}
    print('holdout inventory',audit,'sec',round(time.time()-start,1),flush=True)
    issue_data={}
    results={'precommit':'PHASE3_PRECOMMIT.md','status':'prospectively fixed Imvo-only tests; does not provide second newspaper',
             'selected_1887_only':selection, 'evaluation_audit':audit,'model_validation_reproduction':{}, 'years':{}, 'policy_status':{}}
    for policy in POLICIES:
        model=build_model(train_docs,policy,factors,alphabet)
        results['model_validation_reproduction'][policy]=val_reproduction(model,val_docs,alphabet,policy,prior)
        data=score(model,evaldocs,alphabet,policy)
        for (year,issue),value in data.items():issue_data[(policy,year,issue)]=value
        for year in YEARS:
            results['years'].setdefault(str(year),{})[policy]=aggregate_year(issue_data,year,policy)
        print('finished',policy,'1889',round(results['years']['1889'][policy]['nonmatch'],6),
              '1890',round(results['years']['1890'][policy]['nonmatch'],6),
              'secs',round(time.time()-start,1),flush=True)
        del model
        (OUT/'independent_imvo_partial.json').write_text(json.dumps(results,indent=2))
    for year in YEARS:
        report=bootstrap(issue_data,year,selection)
        a=results['years'][str(year)]['BALANCED'];b=results['years'][str(year)]['HARD']
        report['BALANCED_minus_HARD']={'all':a['all']-b['all'], 'nonmatch':a['nonmatch']-b['nonmatch'],
                                      'match':a['match']-b['match']}
        report['primary_gate']=(report['BALANCED_minus_HARD']['nonmatch']>0 and
                           report['nonmatch']['ci_95'][0]>0) if year==1889 else None
        results['policy_status'][str(year)]=report
        print('year',year,'balanced-hard nonmatch',report['BALANCED_minus_HARD']['nonmatch'],
              'ci',report['nonmatch']['ci_95'],'primary_gate',report['primary_gate'],flush=True)
    (OUT/'independent_imvo_results.json').write_text(json.dumps(results,indent=2))
    print('completed secs',round(time.time()-start,1),flush=True)

if __name__=='__main__':main()
