#!/usr/bin/env python3
"""Fixed 1887 neural training and 1891/1892 heldout Imvo neural replication.

Policy definitions are imported unchanged from pilot_charlm.py. Full experiment
and future-cohort criteria are frozen in NEURAL_PRECOMMIT.md.
"""
import argparse
import collections
import json
import math
import random
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT=Path('/mnt/data/isiXhosa_publication_continuation')
OUT=ROOT/'phase4'
ZIP=Path('/mnt/data/TxtLangFiles1870-1894_TB.zip')
sys.path.insert(0,str(ROOT))
from audit_overlap import (load_pages,index_ngrams,digest,RX_WORD)
from pilot_charlm import (doc_rep,normalize_weight_multipliers,eval_novel_masks,POLICIES)

torch.set_num_threads(4)
N=12
CONTEXT=4
EMBED=32
HIDDEN=96
LR=.003
EPOCHS=3
BATCH=8192
SEEDS=(101,202,303)
FUTURE_RX=re.compile(r'^IMVO_(1891|1892)-(\d\d)-(\d\d)_p\d+\.txt$')


class CharMLP(nn.Module):
    def __init__(self,vocab):
        super().__init__()
        self.embed=nn.Embedding(vocab,EMBED)
        self.h1=nn.Linear(CONTEXT*EMBED,HIDDEN)
        self.h2=nn.Linear(HIDDEN,vocab)

    def forward(self,x):
        return self.h2(F.relu(self.h1(self.embed(x).flatten(1))))


def norm_code(text,alphabet):
    return np.fromiter((alphabet.get(c,alphabet['?']) for c in text), dtype=np.int32,count=len(text))


def samples_for_text(text,alphabet):
    ids=norm_code(text,alphabet)
    if len(ids)<=CONTEXT:
        return np.empty((0,CONTEXT),dtype=np.int32),np.empty(0,dtype=np.int32)
    xx=np.column_stack([ids[i:len(ids)-CONTEXT+i] for i in range(CONTEXT)])
    yy=ids[CONTEXT:]
    assert len(xx)==len(yy)
    return xx,yy


def make_shared_training(pages,train_ix):
    tr=[doc_rep(p,train_ix) for p in pages]
    factors=normalize_weight_multipliers(tr)
    charset=set(''.join(d['text'] for d in tr));charset.add('?')
    alphabet={c:i for i,c in enumerate(sorted(charset))}
    xx=[];yy=[];polweights={policy:[] for policy in POLICIES}
    for d in tr:
        x,y=samples_for_text(d['text'],alphabet)
        xx.append(x);yy.append(y)
        for policy in POLICIES:
            w=np.asarray(d['weights'][policy][CONTEXT:],dtype=np.float32)*factors[policy][d['source']]
            assert len(w)==len(y)
            polweights[policy].append(w)
    xs=np.concatenate(xx);ys=np.concatenate(yy)
    ww={p:np.concatenate(a) for p,a in polweights.items()}
    assert len(xs)==len(ys)
    for p in POLICIES:
        assert len(ww[p])==len(ys)
        # The denominator of the weighted loss is similar, not identically matched,
        # because the first 4 characters of each page do not predict a target.
        assert abs(float(ww[p].mean())-1)<.003,(p,ww[p].mean())
    return xs,ys,ww,alphabet,factors


def read_future():
    pages=[]
    with zipfile.ZipFile(ZIP) as z:
        for name in z.namelist():
            if '/XhTxt/' not in name or '/._' in name or not name.endswith('.txt'):continue
            fn=name.rsplit('/',1)[-1];m=FUTURE_RX.fullmatch(fn)
            if m is None:continue
            year,month,day=m.groups()
            pages.append({'name':fn,'source':'IMVO','issue':'IMVO_'+year+'-'+month+'-'+day,
                          'year':int(year),'tokens':RX_WORD.findall(z.read(name).decode('utf-8','replace').casefold())})
    pages.sort(key=lambda x:x['name'])
    c=collections.Counter(d['year'] for d in pages)
    assert dict(c)=={1891:202,1892:204},c
    return pages


def prep_evaluation(pages,train_ix,alphabet):
    out=[]
    for p in pages:
        words=p['tokens']
        covered=bytearray(len(words))
        for i in range(max(0,len(words)-N+1)):
            links=train_ix.get(digest(words[i:i+N]),())
            if any(issue!=p['issue'] for issue,src in links):covered[i:i+N]=b'\x01'*N
        text=' '.join(words)+' ' if words else ''
        mask=bytearray(len(text))
        k=0
        for word,is_matched in zip(words,covered):
            span=len(word)+1
            if not is_matched:mask[k:k+span]=b'\x01'*span
            k+=span
        assert k==len(text)
        xx,yy=samples_for_text(text,alphabet)
        out.append({'x':xx,'y':yy,'keep':np.frombuffer(bytes(mask[CONTEXT:]),dtype=np.uint8),
                    'issue':p['issue'],'year':p['year'],
                    'oov':sum(c not in alphabet for c in text),'numchars':len(text),'matches':sum(covered)})
    return out


def train_model(x,y,w,policy,seed,epochs=EPOCHS):
    # Same initialization and shuffled order for every policy for a given seed.
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=CharMLP(max(int(y.max())+1,33))
    assert model.h2.out_features==33,('unexpected vocabulary size',model.h2.out_features)
    opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=0)
    xx=torch.from_numpy(x.astype(np.int64,copy=False))
    yy=torch.from_numpy(y.astype(np.int64,copy=False))
    weights=torch.from_numpy(w)
    running=[]
    t=time.perf_counter()
    for epoch in range(epochs):
        order=np.random.default_rng(seed+1000*epoch).permutation(len(yy))
        total=0.;nmass=0.
        for start in range(0,len(order),BATCH):
            idx=order[start:start+BATCH]
            opt.zero_grad(set_to_none=True)
            logits=model(xx[idx]); lossvec=F.cross_entropy(logits,yy[idx],reduction='none')
            loss=(lossvec*weights[idx]).mean()
            loss.backward();opt.step()
            total+=float(loss.detach())*len(idx);nmass+=len(idx)
        avg=total/nmass/math.log(2)
        running.append(avg)
        print('train',policy,'seed',seed,'epoch',epoch+1,'weighted_bpc',round(avg,6),
              'elapsed',round(time.perf_counter()-t,2),flush=True)
    return model,running


@torch.no_grad()
def evaluate(model,docs):
    model.eval()
    outcome={}
    for d in docs:
        n=len(d['y']);allloss=0.;novelloss=0.;nnovel=0
        for start in range(0,n,16384):
            x=torch.from_numpy(d['x'][start:start+16384].astype(np.int64,copy=False))
            y=torch.from_numpy(d['y'][start:start+16384].astype(np.int64,copy=False))
            b=(F.cross_entropy(model(x),y,reduction='none')/math.log(2)).numpy()
            m=d['keep'][start:start+len(b)]
            allloss+=float(b.sum());novelloss+=float(np.dot(b,m));nnovel+=int(m.sum())
        outcome[d['issue']]=outcome.get(d['issue'],np.zeros(4,dtype=np.float64))+np.array([allloss,n,novelloss,nnovel])
    return outcome


def agg_domain(outcome,subset_issues):
    z=np.stack([outcome[i] for i in subset_issues]).sum(axis=0)
    return {'all':float(z[0]/z[1]),'nonmatch':float(z[2]/z[3]),
            'chars_all':int(z[1]),'chars_nonmatch':int(z[3]),'frac_match':float((z[1]-z[3])/z[1])}


def bootstrap(saved,year,policies=('BALANCED','HARD'),B=3000):
    rng=np.random.default_rng(20261025+year)
    # Average losses across matched seeds for every issue, same number of chars.
    rows=sorted(saved[policies[0]][str(SEEDS[0])][str(year)])
    mats={}
    for pol in policies:
        mats[pol]=np.stack([np.mean(np.stack([saved[pol][str(s)][str(year)][issue]
                for s in SEEDS],axis=0),axis=0) for issue in rows])
    a=mats['BALANCED'];b=mats['HARD'];idx=rng.integers(len(rows),size=(B,len(rows)))
    sa=a[idx].sum(axis=1);sb=b[idx].sum(axis=1)
    draws=sa[:,2]/sa[:,3]-sb[:,2]/sb[:,3]
    return {'paired_issue_bootstrap_ci_95':np.quantile(draws,[.025,.975]).tolist(),
            'fraction_positive':float(np.mean(draws>0)),'issue_count':len(rows),
            'point_mean_over_seeds':float(a.sum(axis=0)[2]/a.sum(axis=0)[3]-b.sum(axis=0)[2]/b.sum(axis=0)[3])}


def main(args):
    t=time.time()
    old=load_pages()
    tr=[x for x in old if x['split']=='train'];val=[x for x in old if x['split']=='validation']
    assert len(tr)==218 and len(val)==72
    idx,_=index_ngrams(tr,N)
    train_x,train_y,weights,alphabet,factors=make_shared_training(tr,idx)
    print('train data',train_x.shape,'vocab',len(alphabet),'weight means',
          {x:round(float(weights[x].mean()),5) for x in POLICIES},'secs',round(time.time()-t,2),flush=True)
    val_pages=[{'name':p['name'],'source':p['source'],'issue':p['issue'],'year':1887,
                'tokens':p['tokens']} for p in val]
    valdocs=prep_evaluation(val_pages,idx,alphabet)
    print('val prepared',len(valdocs),flush=True)
    if args.profile:
        model,loss=train_model(train_x,train_y,weights['RAW'],'RAW',101,epochs=1)
        print('PROFILE_VALIDATE',agg_domain(evaluate(model,valdocs),sorted(set(d['issue'] for d in valdocs))),flush=True)
        return
    # Only after protocol and fixed training arrays are saved should new-years be read.
    fut=read_future()
    future_docs=prep_evaluation(fut,idx,alphabet)
    by_year={yr:sorted(set(d['issue'] for d in future_docs if d['year']==yr)) for yr in (1891,1892)}
    print('future pages',len(fut),'issues', {k:len(v) for k,v in by_year.items()},flush=True)
    results={'protocol':'NEURAL_PRECOMMIT.md','model':'embed32 flat 4-char context hidden96 ReLU',
             'training':{'train_pages':len(tr),'val_pages':len(val),'samples':len(train_y),
                         'vocab_size':len(alphabet),'lr':LR,'epochs':EPOCHS,'batch':BATCH,
                         'seeds':SEEDS,'train_effective_mass':{p:float(w.sum()) for p,w in weights.items()}},
             'primary':'1891 Imvo nonmatch BPC BALANCED minus HARD','model_outputs':{},'bootstrap':{}}
    cache={pol:{} for pol in POLICIES}
    for seed in SEEDS:
      for pol in POLICIES:
        model,loss=train_model(train_x,train_y,weights[pol],pol,seed)
        valscore=agg_domain(evaluate(model,valdocs),sorted(set(d['issue'] for d in valdocs)))
        testscores=evaluate(model,future_docs)
        result={str(yr):agg_domain(testscores,issue_list) for yr,issue_list in by_year.items()}
        results['model_outputs'].setdefault(pol,{})[str(seed)]={'train_loss_bpc':loss,'validation_overall':valscore,
                                                                 'test':result}
        for yr in by_year:
            cache[pol].setdefault(str(seed),{})[str(yr)]={issue:testscores[issue].tolist() for issue in by_year[yr]}
        (OUT/'neural_results_partial.json').write_text(json.dumps(results,indent=2))
        print('EVALUATED',pol,seed,result,'elapsed',round(time.time()-t,1),flush=True)
        del model
    for yr in (1891,1892):
        results['bootstrap'][str(yr)]=bootstrap(cache,yr)
    (OUT/'neural_results.json').write_text(json.dumps(results,indent=2))
    (OUT/'neural_issue_scores.json').write_text(json.dumps(cache))
    print('FINISHED',round(time.time()-t,1),'seconds, primary',results['bootstrap']['1891'],
          'secondary',results['bootstrap']['1892'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile',action='store_true',help='Only 1887: single policy, one epoch validation; no future data access.')
    main(parser.parse_args())
