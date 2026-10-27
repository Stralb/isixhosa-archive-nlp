#!/usr/bin/env python3
"""Phase33. Three prospectively frozen sampling policies, BPE, and token-bigram LM.
Requires original user-supplied TxtLangFiles1870-1894_TB.zip. No archive content bundled.
"""
from __future__ import annotations
import argparse, hashlib, json, re, zipfile, csv, time, math
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import pandas as pd
import sentencepiece as spm

PAT = re.compile(r'^(IMVO)_(\d{4})-(\d{2})-(\d{2})_p(\d{3})\.txt$')
WORD = re.compile(r'[a-z]+')
SEEDS = [331,332,333,334,335]
POLICIES = ['P','W','M']
TARGET = 500000
WINDOW_WORDS = 32
VOCAB = 2000
ALPHAMAIN = 10.0
ALPHAS = (1., 10., 100.)

def hash_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def normalize(text):
    return WORD.findall(text.lower())

def parse_archive(path):
    pages={}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if '__MACOSX' in n or not n.endswith('.txt'): continue
            if '/XhTxt/' in n: view='xh'
            elif '/EnTxt/' in n: view='en'
            else: continue
            name=n.rsplit('/',1)[-1]; m=PAT.fullmatch(name)
            if not m:continue
            _,y,month,day,page=m.groups(); date=f'{y}-{month}-{day}'
            rec=pages.setdefault(name,dict(filename=name, date=date,year=int(y),page=int(page)))
            rec[view]=normalize(z.read(n).decode('utf-8-sig',errors='replace'))
    a=sorted(pages.values(),key=lambda r:(r['date'],r['page']))
    assert len(a)>1500, len(a)
    assert all('en' in r and 'xh' in r for r in a)
    return a

def training_units(pages):
    train=[r for r in pages if r['year'] in (1885,1887)]
    assert len(train)==390,len(train)
    units=[]
    for k,r in enumerate(train):
        for view in ('xh','en'):
            if len(r[view])>=16:
                units.append({'pidx':k,'view':view,'words':r[view], 'nwords':len(r[view]),'date':r['date']})
    n=np.zeros((len(train),2),int)
    for u in units:n[u['pidx'],int(u['view']=='en')]=u['nwords']
    assert np.all(n.sum(axis=1)>0)
    p_en=float(np.mean(n[:,1]/n.sum(axis=1)))
    return train,units,n,p_en

def sample_training(train,units,n,p_en,policy,seed,target=TARGET):
    rng=np.random.default_rng(seed*1000 + {'P':1,'W':2,'M':3}[policy])
    by_page=defaultdict(dict)
    for u in units:by_page[u['pidx']][u['view']]=u
    unit_weights=np.array([u['nwords'] for u in units],float);unit_weights/=unit_weights.sum()
    by_view={v:[i for i,u in enumerate(units) if u['view']==v] for v in ('xh','en')}
    probs_by_view={v:unit_weights[idxs]/unit_weights[idxs].sum() for v,idxs in by_view.items()}
    lines=[]; total=0; en_total=0; windows=[]; draw_views=Counter(); selected_pages=Counter()
    while total < target-30:
        if policy=='P':
            pageid=int(rng.integers(0,len(train)))
            c=n[pageid]; view='en' if rng.random()<c[1]/max(c.sum(),1) else 'xh'
            u=by_page[pageid].get(view)
            if u is None:raise ValueError('bad page/view sampling')
        elif policy=='W':
            u=units[int(rng.choice(len(units),p=unit_weights))]
        else:
            view='en' if rng.random()<p_en else 'xh'
            ids=by_view[view]
            u=units[ids[int(rng.choice(len(ids),p=probs_by_view[view]))]]
        w=u['words']; start=int(rng.integers(0,max(1,len(w)-WINDOW_WORDS+1)))
        block=w[start:start+WINDOW_WORDS]
        if not block: continue
        rem=target-total
        keep=[]; nchar=0
        for word in block:
            if nchar+len(word)>rem:break
            keep.append(word);nchar+=len(word)
        if not keep:break
        line=' '.join(keep)
        lines.append(line)
        windows.append(hashlib.blake2b(line.encode('utf8'),digest_size=12).hexdigest())
        total+=nchar
        if u['view']=='en':en_total+=nchar
        draw_views[u['view']]+=1;selected_pages[u['pidx']]+=1
        if total>=target-30:break
    assert target-total<31 and total<=target,(policy,total)
    return lines,{
        'seed':seed,'policy':policy,'training_alpha_chars':int(total),
        'training_en_alpha_chars':int(en_total),'training_en_alpha_share':float(en_total/total),
        'training_draws':len(lines),'unique_training_lines':len(set(windows)),
        'unique_sampled_pages':len(selected_pages),'training_en_windows':draw_views['en'],
        'line_input_sha256':hashlib.sha256(('\n'.join(lines)+'\n').encode()).hexdigest()
    }

def train_model(lines, prefix:Path):
    textpath=prefix.with_suffix('.training.tmp')
    textpath.write_text('\n'.join(lines)+'\n',encoding='utf8')
    try:
        spm.SentencePieceTrainer.train(input=str(textpath),model_prefix=str(prefix),vocab_size=VOCAB,
             model_type='bpe', character_coverage=1.,hard_vocab_limit=False,
             normalization_rule_name='identity', bos_id=-1, eos_id=-1, pad_id=-1,unk_id=0,
             num_threads=1, max_sentence_length=4096, input_sentence_size=0,shuffle_input_sentence=False,
             split_by_whitespace=True, add_dummy_prefix=True, remove_extra_whitespaces=True)
    finally:
        textpath.unlink(missing_ok=True)
    sp=spm.SentencePieceProcessor(model_file=str(prefix)+'.model')
    assert sp.vocab_size()==VOCAB,(sp.vocab_size(),prefix)
    return sp

def eval_units(pages):
    train=[];held=[]
    for r in pages:
        if r['year'] not in (1885,1887) and not (1888<=r['year']<=1894):continue
        for view in ('xh','en'):
            w=r[view]
            if len(w)<40:continue
            words=w[:128]
            x=' '.join(words);ch=sum(map(len,words))
            a={'filename':r['filename'],'date':r['date'],'year':r['year'],'page':r['page'],
               'view':view,'text':x,'chars':ch,'words':len(words)}
            (train if r['year'] in (1885,1887) else held).append(a)
    assert all(a['year'] in (1885,1887) for a in train)
    assert all(a['year']>=1888 for a in held)
    return train,held

class UniBackoffLM:
    def __init__(self,sp,units):
        self.sp=sp;self.big=Counter();self.first=Counter();self.uni=Counter();self.ntokens=0
        self.vocab=sp.vocab_size()
        for r in units:
            ids=sp.encode(r['text'],out_type=int)
            p=-1
            for t in ids:
                self.big[(p,t)]+=1;self.first[p]+=1;self.uni[t]+=1;self.ntokens+=1;p=t
        assert self.ntokens>0
    def bits(self,ids,alpha):
        p=-1;logloss=0.0
        den=self.ntokens+0.1*self.vocab
        for t in ids:
            pr=(self.big[(p,t)]+alpha*(self.uni[t]+0.1)/den)/(self.first[p]+alpha)
            assert pr>0
            logloss-=math.log2(pr)
            p=t
        return logloss

def produce_scores(sp,seed,policy,train_units,held_units):
    lm=UniBackoffLM(sp,train_units)
    scores=[]
    for a in held_units:
        ids=sp.encode(a['text'],out_type=int)
        d={k:a[k] for k in ('filename','date','year','page','view','chars','words')}
        d.update(seed=seed,policy=policy,pieces=len(ids),unknown=sum(t==sp.unk_id() for t in ids),
                lm_bits_1=lm.bits(ids,1.),lm_bits_10=lm.bits(ids,10.),lm_bits_100=lm.bits(ids,100.))
        scores.append(d)
    return scores,lm.ntokens

def paired_boot(df,view,cohort,a,b,metric,seed=33001,B=2000,by_year=False):
    x=df[(df['view']==view)&(df['cohort']==cohort)]
    seeds=sorted(x.seed.unique())
    clusters=sorted(x.year.unique()) if by_year else sorted(x.date.unique())
    key='year' if by_year else 'date'
    idx=pd.MultiIndex.from_product([clusters,seeds],names=[key,'seed'])
    def extract(policy):
        agg=(x[x.policy==policy].groupby([key,'seed'])[['chars',metric]].sum()
               .reindex(idx,fill_value=0).to_numpy().reshape(len(clusters),len(seeds),2))
        return agg
    aa=extract(a);bb=extract(b)
    assert np.array_equal(aa[:,:,0],bb[:,:,0])
    den=aa[:,:,0]; delta=aa[:,:,1]-bb[:,:,1]
    factor=100.0 if metric=='pieces' else 1.0
    point=float(np.mean(factor*delta.sum(axis=0)/den.sum(axis=0)))
    rng=np.random.default_rng(seed)
    # Multinomial multiplicities are exactly the distribution of bootstrap
    # resampling whole issues with replacement, but avoid slow Python loops.
    weights=rng.multinomial(len(clusters), np.full(len(clusters),1./len(clusters)),size=B)
    numer=weights@delta;denom=weights@den
    assert np.all(denom>0)
    dist=np.mean(factor*numer/denom,axis=1)
    return dict(metric=metric,view=view,cohort=cohort,contrast=f'{a}-{b}',estimate=point,
        interval_low=float(np.quantile(dist,.025)),interval_high=float(np.quantile(dist,.975)),
        replicates=B,clusters=len(clusters),seed_count=len(seeds),resample_unit=key)

def summarize(held, out, audits):
    df=pd.DataFrame(held)
    df['cohort']=np.where(df.year==1888,'heldout_1888','later_1889_1894')
    df.to_csv(out/'page_scores.csv',index=False)
    # each paired page has exactly one score for every policy x seed
    expected=len(df)/15
    assert expected.is_integer()
    assert df.groupby(['filename','view']).size().eq(15).all()
    assert all(len(x)==expected for _,x in df.groupby(['seed','policy']))
    results=[]
    for (cohort,view,policy,seed),g in df.groupby(['cohort','view','policy','seed'],sort=True):
        results.append(dict(cohort=cohort,view=view,policy=policy,seed=seed,
               issues=g.date.nunique(),pages=len(g),alphabetic_chars=int(g.chars.sum()),
               pieces_per_100=100*g.pieces.sum()/g.chars.sum(),
               unknown_per_100=100*g.unknown.sum()/g.chars.sum(),
               bpc_1=g.lm_bits_1.sum()/g.chars.sum(),
               bpc_10=g.lm_bits_10.sum()/g.chars.sum(),
               bpc_100=g.lm_bits_100.sum()/g.chars.sum()))
    ss=pd.DataFrame(results);ss.to_csv(out/'per_seed_summary.csv',index=False)
    contrasts=[]
    for cohort in ('heldout_1888','later_1889_1894'):
        for v in ('en','xh'):
            for a,b in [('P','W'),('P','M'),('M','W')]:
                for met in ('pieces','lm_bits_10','lm_bits_1','lm_bits_100'):
                    contrasts.append(paired_boot(df,v,cohort,a,b,met,seed=33001+{'en':1,'xh':2}[v]+100*(cohort=='later_1889_1894')+{'pieces':0,'lm_bits_10':10,'lm_bits_1':20,'lm_bits_100':30}[met]))
                if cohort=='later_1889_1894' and a=='P' and b=='W':
                    for met in ('pieces','lm_bits_10'):
                        contrasts.append(paired_boot(df,v,cohort,a,b,met,seed=33500+int(met!='pieces'),by_year=True))
    cc=pd.DataFrame(contrasts);cc.to_csv(out/'paired_contrasts.csv',index=False)
    aa=pd.DataFrame(audits);aa.to_csv(out/'training_audit.csv',index=False)
    summaries=ss.groupby(['cohort','view','policy'],as_index=False)[['pieces_per_100','bpc_10','unknown_per_100']].mean()
    summaries.to_csv(out/'summary.csv',index=False)
    def get(cohort,v,m):
        return cc[(cc.cohort==cohort)&(cc.view==v)&(cc.contrast=='P-W')&(cc.metric==m)&(cc.resample_unit=='date')].iloc[0].to_dict()
    share=float(aa.groupby('policy').training_en_alpha_share.mean().loc['W']-aa.groupby('policy').training_en_alpha_share.mean().loc['P'])
    encost=get('heldout_1888','en','pieces');enbpc=get('heldout_1888','en','lm_bits_10')
    replication_c=get('later_1889_1894','en','pieces');replication_b=get('later_1889_1894','en','lm_bits_10')
    def dcast(x):return {k:float(v) if isinstance(v,(np.floating,np.integer)) else v for k,v in x.items()}
    decisions={
      'H33a':{'status':'SUPPORTED' if share>=.04 else 'NOT_SUPPORTED', 'word_minus_page_en_character_share':share},
      'H33b':{'status':'SUPPORTED' if encost['estimate']>=.30 and encost['interval_low']>0 and replication_c['estimate']>0 else 'NOT_SUPPORTED',
               '1888':dcast(encost),'later':dcast(replication_c)},
      'H33c':{'status':'SUPPORTED' if enbpc['estimate']>=.01 and enbpc['interval_low']>0 and replication_b['estimate']>0 else 'NOT_SUPPORTED',
               '1888':dcast(enbpc),'later':dcast(replication_b)}}
    (out/'RESULTS.json').write_text(json.dumps({'hypotheses':decisions,'sampling':aa.to_dict('records'),
                  'training_models':len(aa),'eval_page_view_units':int(expected),
                  'LM_method':'unigram-backoff smoothed token bigram, alpha=10',
                  'note':'Archive-assigned language views, not expert validated.'},indent=2,default=str)+'\n')
    return decisions

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--zip',required=True);ap.add_argument('--out',required=True); args=ap.parse_args()
    out=Path(args.out);out.mkdir(exist_ok=True,parents=True);(out/'models').mkdir(exist_ok=True)
    st=time.time()
    pages=parse_archive(args.zip);train,units,n,p_en=training_units(pages)
    trlm,held=eval_units(pages)
    print('ARCHIVE PAGES',len(pages),'TRAIN PAGES',len(train),'LM TRAIN UNITS',len(trlm),'HELD UNITS',len(held),'P EN expected',p_en,flush=True)
    assert len(set(r['filename'] for r in trlm)&set(r['filename'] for r in held))==0
    assert len(held)>1000
    audits=[];saved=[]
    for seed in SEEDS:
        for policy in POLICIES:
            prefix=out/'models'/f'bpe_{policy}_{seed}'
            lines,audit=sample_training(train,units,n,p_en,policy,seed)
            sp=train_model(lines,prefix)
            audit.update(model_sha256=hash_file(str(prefix)+'.model'),vocab_size=sp.vocab_size(),
                         training_languages='IMVO_1885_and_1887',expected_P_en_word_share=p_en)
            audits.append(audit)
            scores,lm_tokens=produce_scores(sp,seed,policy,trlm,held)
            audit['LM_train_tokens']=int(lm_tokens)
            saved.extend(scores)
            print('MODEL',policy,seed,'share',round(audit['training_en_alpha_share'],4),'tokens',lm_tokens,'scores',len(scores),'elapsed',round(time.time()-st,1),flush=True)
    decisions=summarize(saved,out,audits)
    print('DECISIONS',json.dumps(decisions,indent=2),flush=True)
    print('TIME',time.time()-st,flush=True)
if __name__=='__main__':main()
