#!/usr/bin/env python3
"""Frozen four-arm sampler for Phase 2. Reads only designated training years.

Factors: view-allocation P/W crossed with within-view page-selection P/W.
PP = original Page-uniform; PW = earlier 'M'; WW = original Word-proportional;
WP = previously missing arm. No test or validation material is used for sampling.
"""
from __future__ import annotations
import hashlib
import numpy as np
from collections import defaultdict
from audit_phase1 import K, WINDOW, enumerate_units

ARMS=('PP','PW','WP','WW')

def build_pool(train, exclude_reprints=False, common_support=False):
    """Return eligible starts for each page/view. Duplicate labels use TRAIN issues only."""
    units, counts=enumerate_units(train)
    owners=defaultdict(set)
    if exclude_reprints or common_support:
        for u in units:
            w=u['words']
            for i in range(len(w)-K+1):owners[tuple(w[i:i+K])].add(u['issue'])
        repeated={x for x,issues in owners.items() if len(issues)>1}
    else:repeated=set()
    eligible=[]
    for u in units:
        w=u['words'];n=len(w);starts=max(n-WINDOW+1,1)
        if exclude_reprints or common_support:
            flags=[tuple(w[i:i+K]) in repeated for i in range(n-K+1)]
            clean=np.array([i for i in range(starts) if not any(flags[i:min(n,i+WINDOW)-K+1])],dtype=np.int32)
            if not len(clean):continue
            possible=clean if exclude_reprints else np.arange(starts,dtype=np.int32)
        else:possible=np.arange(starts,dtype=np.int32)
        eligible.append({**u,'starts':possible})
    return eligible

def distributions(pool,arm):
    assert arm in ARMS
    # P view allocation is the mean eligible page's proportion of view words;
    # W is the aggregate ratio of word counts. Conditional page P gives
    # q(page|view) proportional to its share of that page's eligible word count.
    bypage=defaultdict(lambda:{'xh':0,'en':0})
    for u in pool:bypage[u['file']][u['view']]=u['nwords']
    pvals=list(bypage.values());n=len(pvals)
    page_view_p=sum(x['en']/(x['en']+x['xh']) for x in pvals)/n
    word_view_p=sum(x['en'] for x in pvals)/sum(x['en']+x['xh'] for x in pvals)
    p_en=page_view_p if arm[0]=='P' else word_view_p
    ids={v:[i for i,u in enumerate(pool) if u['view']==v] for v in ('en','xh')}
    prob={}
    for v in ('en','xh'):
        if arm[1]=='P':
            wt=np.array([pool[i]['nwords']/(bypage[pool[i]['file']]['en']+bypage[pool[i]['file']]['xh']) for i in ids[v]],float)
        else:wt=np.array([pool[i]['nwords'] for i in ids[v]],float)
        assert len(wt)>0 and wt.sum()>0
        prob[v]=wt/wt.sum()
    return ids,prob,p_en

def draw(pool,arm,seed,budget=500000):
    ids,prob,p_en=distributions(pool,arm)
    rng=np.random.default_rng(int(seed)*1000 + {'PP':11,'PW':12,'WP':13,'WW':14}[arm])
    lines=[];total=0;en_total=0;windows=set();sample_issues=set();views=defaultdict(int)
    while total<budget-30:
        view='en' if rng.random()<p_en else 'xh'
        u=pool[ids[view][int(rng.choice(len(ids[view]),p=prob[view]))]]
        start=int(u['starts'][int(rng.integers(len(u['starts'])))])
        block=u['words'][start:start+WINDOW]
        remaining=budget-total;keep=[];nchars=0
        for word in block:
            if nchars+len(word)>remaining:break
            keep.append(word);nchars+=len(word)
        if not keep:break
        text=' '.join(keep);lines.append(text)
        total+=nchars;en_total+=nchars*(view=='en')
        windows.add(hashlib.sha256(text.encode()).digest());sample_issues.add(u['issue']);views[view]+=1
    assert budget-30<=total<=budget and len(lines)>10
    metadata={'arm':arm,'seed':seed,'budget_target':budget,'actual_alpha_chars':total,
       'en_alpha_chars':en_total,'en_alpha_share':en_total/total,'views_drawn_en':views['en'],
       'views_drawn_xh':views['xh'],'unique_lines':len(windows),'lines':len(lines),
       'source_issues_represented':len(sample_issues),
       'sample_hash_sha256':hashlib.sha256(('\n'.join(lines)+'\n').encode()).hexdigest()}
    return lines,metadata
