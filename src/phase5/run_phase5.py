#!/usr/bin/env python3
"""Phase 5: F5 preregistered LM-BPC factorial downstream gate.

No tokenizer refitting. Smoothing alpha=100 locked by Phase2 train/validation.
Scores same Phase3 held-out samples in identical order and documents for 40 SP models.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import sentencepiece as spm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'Phase2'/'scripts'))
sys.path.insert(0, str(ROOT/'Phase3'/'scripts'))
from run_phase2 import build_lm_docs, SmoothedBigram, sha as hashlib_str
from audit_phase1 import parse
from run_phase3 import (audit_models, load_heldout, digest, csvread, writecsv, FAMILIES, ARMS, SEEDS, VIEWS, PERIODS,
    N_BOOT, BOOT_SEED, contrasts)

FROZEN_SHA='8575dc1ae52a22391318933bc5d4c8cee8777c15645a9eed13bbe34bb15583f8'
ALPHA=100.0
OUT_COLS=['family','arm','seed','filename','view','issue','year','period','words','chars','text_sha256',
          'pieces','unknown','bits','bpc']

def safe_write_csv(path, rows, fields):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.incomplete')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    tmp.replace(path)

def check_training(root, archive):
    train, val, _, reads = parse(archive)
    assert reads['heldout_view_content_reads']==0
    docs=build_lm_docs(train)
    frozen=csvread(root/'Phase2'/'outputs'/'lm_training_manifest.csv')
    assert len(frozen)==len(docs)==545
    for a,b in zip(docs,frozen):
        for k in ('filename','issue','view','chars','text_sha256'):
            assert str(a[k])==str(b[k]),('LM training manifest mismatch',k,a,b)
    assert set(d['issue'] for d in docs).isdisjoint({r['issue'] for r in val})
    assert len({d['issue'] for d in docs})>=60
    return docs,train,val

def check_pages(root, archive):
    pages=load_heldout(archive)
    frozen=csvread(root/'Phase3'/'outputs'/'heldout_pageview_manifest.csv')
    assert len(pages)==len(frozen)==2493
    for a,b in zip(pages,frozen):
        for k in ('filename','view','year','issue','period','words','total_view_words','chars','text_sha256'):
            assert str(a[k])==str(b[k]),('test manifest mismatch',k,a,b)
    assert len({p['issue'] for p in pages})==358
    assert all(p['year']>=1888 for p in pages)
    return pages

def score_one(modelname,model_info,docs,pages,phase3_rows):
    family,arm,seed=modelname.split('_');seed=int(seed)
    sp=spm.SentencePieceProcessor(model_file=str(model_info['path']))
    lm=SmoothedBigram(sp,docs)
    assert lm.ntokens==int(model_info['source']['LM_training_tokens']), (modelname,lm.ntokens)
    assert sp.vocab_size()==2000
    scores=[]
    for r in pages:
        ids=sp.encode(r['text'],out_type=int)
        ref=phase3_rows[(family,arm,seed,r['filename'],r['view'])]
        assert len(ids)==int(ref['pieces']),('Phase 3 encoding changed',modelname,r['filename'],r['view'])
        unknown=sum(t==sp.unk_id() for t in ids)
        assert unknown==int(ref['unknown'])
        bits=lm.bits(ids,ALPHA)
        scores.append(dict(family=family,arm=arm,seed=seed,filename=r['filename'],view=r['view'],issue=r['issue'],
                           year=r['year'],period=r['period'],words=r['words'],chars=r['chars'],
                           text_sha256=r['text_sha256'],pieces=len(ids),unknown=unknown,bits=format(bits,'.17g'),
                           bpc=format(bits/r['chars'],'.17g')))
    return scores,{'model':modelname,'LM_tokens':lm.ntokens,'LM_docs':len(docs),
                   'model_sha256':digest(model_info['path']),'heldout_rows':len(scores),
                   'heldout_bits':sum(float(x['bits']) for x in scores),'heldout_unknown':sum(x['unknown'] for x in scores)}

def group_metrics(rows):
    df=pd.DataFrame(rows)
    for k in ('seed','chars','year','pieces','unknown'):df[k]=df[k].astype(int)
    df['bits']=df['bits'].astype(float)
    x=df.groupby(['family','period','view','seed','arm'],sort=True).agg(
        bits=('bits','sum'),chars=('chars','sum'),pieces=('pieces','sum'),unknown=('unknown','sum'),
        page_views=('filename','size'),issues=('issue','nunique')).reset_index()
    x['bpc']=x['bits']/x['chars']
    x['pieces_per_100']=100*x['pieces']/x['chars']
    return df,x

def bootstrap(df):
    """Same whole-issue paired-resampling rules as Phase3; numerator in bits not pieces."""
    results=[];seed_rows=[];year_rows=[];meta=[]
    for period in PERIODS:
        subset=df[df.period==period]
        issues=sorted(subset.issue.unique());n=len(issues)
        assert n== (52 if period=='1888' else 306)
        # Phase 3 fixed paired design. Each row in wt is an entire issue resampling draw.
        rng=np.random.default_rng(BOOT_SEED+(period!='1888'))
        wt=rng.multinomial(n,[1./n]*n,size=N_BOOT).astype(float)
        meta.append(dict(period=period,issue_clusters=n,bootstraps=N_BOOT,seed=BOOT_SEED+(period!='1888'),
                        family_and_arm_bootstrap_shared=True))
        if period=='1889-94':
            years=sorted(subset.year.unique());assert years==list(range(1889,1895))
            yrng=np.random.default_rng(BOOT_SEED+100)
            yearwt=yrng.multinomial(6,[1/6]*6,size=N_BOOT).astype(float)
        for fam in FAMILIES:
            for view in VIEWS:
                dfq=subset[(subset.family==fam)&(subset.view==view)]
                assert set(zip(dfq.arm,dfq.seed))=={(a,s) for a in ARMS for s in SEEDS}
                charidx=dfq[(dfq.arm=='PP')&(dfq.seed==SEEDS[0])].groupby('issue').chars.sum()
                chars=np.array([float(charidx.get(k,0)) for k in issues],dtype=float)
                assert chars.sum()>0
                bitsidx=dfq.groupby(['arm','seed','issue']).bits.sum()
                bitarr=np.zeros((len(ARMS),len(SEEDS),n),dtype=float)
                for i,a in enumerate(ARMS):
                    for j,s in enumerate(SEEDS):
                        bitarr[i,j]=[float(bitsidx.get((a,s,issue),0)) for issue in issues]
                point=bitarr.sum(axis=2)/chars.sum()
                dens=wt@chars
                assert (dens>0).all()
                sampled=(wt@bitarr.reshape(len(ARMS)*len(SEEDS),n).T).reshape(N_BOOT,len(ARMS),len(SEEDS))
                sampled=sampled/dens[:,None,None]
                pr=contrasts({a:point[i] for i,a in enumerate(ARMS)})
                boot=contrasts({a:sampled[:,i,:] for i,a in enumerate(ARMS)})
                for contrast,effects in pr.items():
                    effects=np.asarray(effects)
                    b=np.asarray(boot[contrast]).mean(axis=1)
                    l,h=np.quantile(b,[.025,.975])
                    row=dict(family=fam,period=period,view=view,contrast=contrast,
                        mean=float(effects.mean()),ci_low=float(l),ci_high=float(h),
                        seeds=5,issues=n,page_views=len(dfq)//20,
                        seed_min=float(effects.min()),seed_max=float(effects.max()),
                        seeds_negative=int((effects<0).sum()),seeds_positive=int((effects>0).sum()))
                    results.append(row)
                    for s,effect in zip(SEEDS,effects):seed_rows.append(dict(family=fam,period=period,view=view,
                                                      seed=s,contrast=contrast,value=float(effect)))
                if period=='1889-94':
                    yearpos={y:i for i,y in enumerate(years)}
                    chyr=np.zeros(6,dtype=float)
                    biyr=np.zeros((len(ARMS),len(SEEDS),6),dtype=float)
                    for ix,issue in enumerate(issues):
                        yi=yearpos[int(issue[5:9])]
                        chyr[yi]+=chars[ix]
                        biyr[:,:,yi]+=bitarr[:,:,ix]
                    sy=(yearwt@biyr.reshape(20,6).T).reshape(N_BOOT,4,5)
                    dyear=yearwt@chyr
                    assert (dyear>0).all()
                    bs=contrasts({a:sy[:,i,:]/dyear[:,None] for i,a in enumerate(ARMS)})
                    for contrast,boots in bs.items():
                        l,h=np.quantile(boots.mean(axis=1),[.025,.975])
                        rr=next(x for x in results if x['family']==fam and x['period']==period and x['view']==view and x['contrast']==contrast)
                        rr['year_cluster_ci_low']=float(l)
                        rr['year_cluster_ci_high']=float(h)
                    for yi,y in enumerate(years):
                        yearly=contrasts({a:biyr[i,:,yi]/chyr[yi] for i,a in enumerate(ARMS)})
                        for c,x in yearly.items():
                            year_rows.append(dict(family=fam,view=view,year=y,contrast=c,
                                                  mean=float(np.mean(x)),seeds_negative=int((x<0).sum()),
                                                  seeds_positive=int((x>0).sum())))
    return results,seed_rows,year_rows,meta

def duplicate_sensitivity(root,df):
    """Conservative archival train 13-token overlap annotation already frozen Phase4."""
    over=csvread(root/'Phase4'/'outputs'/'train_test_overlap_diagnostic.csv')
    key={(r['filename'],r['view']):r for r in over}
    assert len(key)==len(over)==2493
    pagepairs=df[['filename','view','text_sha256','period']].drop_duplicates()
    assert len(pagepairs)==2493
    joined=pagepairs.copy()
    joined['overlap_13grams']=joined.apply(lambda r:int(key[(r['filename'],r['view'])]['train_overlap_13grams']),axis=1)
    joined['clean_from_train_13gram']=joined['overlap_13grams']==0
    # Each clean pageview remains equally represented across family/arm/seed.
    clean={(r.filename,r.view) for r in joined.itertuples() if r.clean_from_train_13gram}
    dfc=df[df.apply(lambda r:(r['filename'],r['view']) in clean,axis=1)]
    pooled=[]
    for (fam,period,view),g in dfc.groupby(['family','period','view']):
        sums=g.groupby(['arm','seed']).agg(bits=('bits','sum'),chars=('chars','sum'))
        e={a:np.array([sums.loc[(a,s),'bits']/sums.loc[(a,s),'chars'] for s in SEEDS]) for a in ARMS}
        contrasts_clean=contrasts(e)
        for c,effect in contrasts_clean.items():
            pooled.append(dict(family=fam,period=period,view=view,contrast=c,mean=float(np.mean(effect)),
                            min_seed=float(np.min(effect)),max_seed=float(np.max(effect)),
                            page_views=len(g)//20,issues=g.issue.nunique()))
    return joined,pooled

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--archive',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--limit',type=int,default=0);args=ap.parse_args()
    root=Path(args.root);out=Path(args.output);out.mkdir(parents=True,exist_ok=True);(out/'model_rows').mkdir(exist_ok=True)
    assert digest(args.archive)=='375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5'
    assert digest(root/'Phase1'/'FROZEN_PROTOCOL.md')=='bc443d883fa1c2b8eb09ced5f87d8f65b352d55512ef2c594b1b23e64d496bf4'
    assert digest(root/'Phase5'/'PRECOMMIT.md')==FROZEN_SHA
    models,sel=audit_models(root)
    assert sel['selected_alpha']==ALPHA and sel['selected_before_1888_1894_factorial_evaluation']
    docs,train,val=check_training(root,args.archive)
    pages=check_pages(root,args.archive)
    assert set(p['issue'] for p in pages).isdisjoint(set(d['issue'] for d in docs))
    phase3_rows=csvread(root/'Phase3'/'outputs'/'heldout_scores.csv')
    ref={(r['family'],r['arm'],int(r['seed']),r['filename'],r['view']):r for r in phase3_rows}
    assert len(ref)==len(phase3_rows)==len(models)*len(pages)==99720
    audit=[];total=0;t0=time.time()
    for name,md in sorted(models.items()):
        outpath=out/'model_rows'/f'{name}.csv';auditpath=out/'model_rows'/f'{name}.audit.json'
        if outpath.exists() and auditpath.exists():
            a=json.loads(auditpath.read_text())
            assert digest(outpath)==a['scores_sha256'] and a['model_sha256']==digest(md['path'])
            audit.append(a);total+=1;print('REUSE',name,total,'/',len(models),round(time.time()-t0,2),flush=True)
        else:
            scores,a=score_one(name,md,docs,pages,ref)
            safe_write_csv(outpath,scores,OUT_COLS)
            a['scores_sha256']=digest(outpath)
            auditpath.write_text(json.dumps(a,indent=2,sort_keys=True)+'\n')
            audit.append(a);total+=1;print('SCORED',name,total,'/',len(models),round(time.time()-t0,2),flush=True)
        if args.limit and total>=args.limit:break
    if total<len(models):print('CHECKPOINT_ONLY',total);return
    rows=[]
    for a in sorted(audit,key=lambda x:x['model']):rows+=csvread(out/'model_rows'/f'{a["model"]}.csv')
    assert len(rows)==99720
    safe_write_csv(out/'heldout_lm_scores.csv',rows,OUT_COLS)
    df,metrics=group_metrics(rows)
    metrics.to_csv(out/'per_model_view_metrics.csv',index=False,float_format='%.17g')
    contr,seedrows,yearrows,meta=bootstrap(df)
    safe_write_csv(out/'bpc_factorial_contrasts.csv',contr,list(dict.fromkeys(k for r in contr for k in r)))
    safe_write_csv(out/'per_seed_effects.csv',seedrows,list(seedrows[0]))
    safe_write_csv(out/'per_year_effects.csv',yearrows,list(yearrows[0]))
    safe_write_csv(out/'bootstrap_design.csv',meta,list(meta[0]))
    overlap,clean=duplicate_sensitivity(root,df)
    overlap.to_csv(out/'heldout_repetition_overlap.csv',index=False)
    safe_write_csv(out/'no_training_13gram_overlap_contrasts.csv',clean,list(clean[0]))
    lk={(r['family'],r['period'],r['view'],r['contrast']):r for r in contr}
    main=lk[('bpe','1888','en','allocation')]
    late=lk[('bpe','1889-94','en','allocation')]
    ok=main['mean']<=-0.01 and main['ci_high']<0 and late['mean']<0
    decision={'F5':('SUPPORTED' if ok else 'NOT_SUPPORTED'),
              'preregistered_requirements':{'bpe_1888_en_allocation_at_most_negative_0_01':main['mean']<=-0.01,
                'bpe_1888_ci_high_below_zero':main['ci_high']<0,'later_same_negative_sign':late['mean']<0},
              '1888':main,'1889_94':late,'alpha':ALPHA,'full_support_required':True,
              'interpretation':'smoothed count bigram LM only; no neural, social harm or cultural inference'}
    (out/'F5_DECISION.json').write_text(json.dumps(decision,indent=2,sort_keys=True)+'\n')
    man={'phase5_completed':True,'archive_sha256':digest(args.archive),'protocol_sha256':digest(root/'Phase1'/'FROZEN_PROTOCOL.md'),
       'phase5_precommit_sha256':digest(root/'Phase5'/'PRECOMMIT.md'),
       'phase2_selection_sha256':digest(root/'Phase2'/'outputs'/'PHASE2_SELECTION_FREEZE.json'),
       'fixed_alpha':ALPHA,'LM_train_docs':len(docs),'LM_train_view_counts':dict(collections.Counter(d['view'] for d in docs)),
       'LM_train_doc_sha256':hashlib.sha256(('\n'.join(d['text_sha256'] for d in docs)+'\n').encode()).hexdigest(),
       'models':len(models),'model_scores':len(rows),'heldout_pageviews':len(pages),
       'heldout_issues':len({p['issue'] for p in pages}),'unknown_tokens':sum(int(r['unknown']) for r in rows),
       'periods':{per:dict(pageviews=sum(p['period']==per for p in pages),issues=len({p['issue'] for p in pages if p['period']==per})) for per in PERIODS},
       'train_test_13gram_clean_pages':int(overlap['clean_from_train_13gram'].sum()),
       'comparisons':len(contr),'seed_effects':len(seedrows),'yearly_effects':len(yearrows),'decision':decision,
       'analyst_reuse_caveat':'1888-1894 newspaper test years studied in Phases 26-33 and Excellence Sprint 3-4; not untouched external evidence'}
    (out/'PHASE5_MANIFEST.json').write_text(json.dumps(man,indent=2,sort_keys=True)+'\n')
    print('F5_FINAL_DECISION',json.dumps(decision,indent=2),flush=True)
    print('DONE',time.time()-t0,flush=True)

if __name__=='__main__':main()
