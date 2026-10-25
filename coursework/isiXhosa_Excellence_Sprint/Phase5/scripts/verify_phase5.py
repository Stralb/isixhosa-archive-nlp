#!/usr/bin/env python3
"""Independent Phase5 integrity checker (does NOT use phase2 SmoothedBigram/phase5 scorer)."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, re, zipfile, collections
from pathlib import Path
import numpy as np
import pandas as pd
import sentencepiece as spm

R=re.compile(r'^(IMVO)_(\d{4})-(\d{2})-(\d{2})_p\d{3}\.txt$')
WORD=re.compile(r'[a-z]+')
ARMS=['PP','PW','WP','WW'];FAM=['bpe','unigram'];SEEDS=[441,442,443,444,445];PERIODS=['1888','1889-94'];VIEWS=['en','xh']

def sha(b):return hashlib.sha256(b).hexdigest()
def digest(path):return sha(Path(path).read_bytes())
def readcsv(p):
    with open(p,newline='',encoding='utf8') as f:return list(csv.DictReader(f))

def get_source_texts(archive):
    tr={};ev={};dates=set()
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            parts=entry.filename.split('/')
            if len(parts)<2 or parts[-2] not in ('EnTxt','XhTxt'):continue
            mt=R.fullmatch(parts[-1])
            if not mt:continue
            y,m,d=map(int,mt.groups()[1:]);name=parts[-1];view='en' if parts[-2]=='EnTxt' else 'xh'
            if not ((y==1885 or (y==1887 and m<=9)) or (1888<=y<=1894)):continue
            words=WORD.findall(z.read(entry).decode('utf-8-sig',errors='replace').lower())
            issue=f'IMVO_{y:04d}-{m:02d}-{d:02d}'
            if (y==1885 or (y==1887 and m<=9)):
                key=(name,view)
                assert key not in tr
                tr[key]=dict(words=words,issue=issue,year=y,filename=name,view=view)
            elif len(words)>=40:
                text=' '.join(words[:128]);key=(name,view)
                assert key not in ev
                ev[key]=dict(text=text,chars=sum(map(len,words[:128])),issue=issue,year=y,filename=name,view=view,
                        period='1888' if y==1888 else '1889-94',text_sha256=sha(text.encode()))
    docs=[]
    # Independent regeneration: same filename ordering and fixed en then xh view ordering.
    for name in sorted({x[0] for x in tr}):
        assert (name,'en') in tr and (name,'xh') in tr
        for view in ('en','xh'):
            t=tr[(name,view)]
            if len(t['words'])<40:continue
            s=' '.join(t['words'][:128]);docs.append(dict(text=s,filename=name,issue=t['issue'],view=view,
                       chars=sum(map(len,t['words'][:128])),text_sha256=sha(s.encode())))
    pages=[ev[k] for k in sorted(ev)]
    assert len(docs)==545 and len(pages)==2493
    assert {d['issue'] for d in docs}.isdisjoint({p['issue'] for p in pages})
    return docs,pages

def run_one(family,arm,seed,source_docs,source_pages,root,out):
    name=f'{family}_{arm}_{seed}'
    mp=root/'Phase2'/'outputs'/'models'/f'{name}.model'
    meta={x['model_name']:x for x in readcsv(root/'Phase2'/'outputs'/'model_training_manifest.csv')}
    assert digest(mp)==meta[name]['model_sha256']
    assert digest(mp.with_suffix('.vocab'))==meta[name]['vocab_sha256']
    assert int(meta[name]['model_vocab_size'])==2000
    p=spm.SentencePieceProcessor(model_file=str(mp));v=p.vocab_size()
    pair=collections.defaultdict(int)
    pred=collections.defaultdict(int)
    freq=collections.defaultdict(int)
    n=0
    # Independently written count implementation: enumerate ngrams using adjacent zip.
    for doc in source_docs:
        seq=p.encode(doc['text'],out_type=int)
        n+=len(seq)
        for tok in seq:freq[tok]+=1
        for previous,tok in zip([-1]+seq[:-1],seq):
            pair[(previous,tok)]+=1
            pred[previous]+=1
    assert n==int(meta[name]['LM_training_tokens'])
    assert sum(freq.values())==n and sum(pred.values())==n and sum(pair.values())==n
    csvpath=out/'model_rows'/f'{name}.csv'
    records=readcsv(csvpath)
    assert len(records)==len(source_pages)
    max_abs_bits=max_abs_bpc=0.0
    max_abs_pieces=0;unknowns=0
    den=n+0.1*v
    for actual,source in zip(records,source_pages):
        assert actual['filename']==source['filename'] and actual['view']==source['view']
        assert actual['issue']==source['issue'] and actual['period']==source['period']
        assert actual['text_sha256']==source['text_sha256']
        assert int(actual['chars'])==source['chars']
        ids=p.encode(source['text'],out_type=int)
        assert len(ids)==int(actual['pieces'])
        bad=sum(t==p.unk_id() for t in ids)
        assert bad==int(actual['unknown'])
        unknowns+=bad
        # Read-only, independent predictive formula; reset -1 at each new page-view.
        cost=math.fsum(-math.log2((pair[(last,t)]+100*((freq[t]+.1)/den))/(pred[last]+100.))
                        for last,t in zip([-1]+ids[:-1],ids))
        max_abs_bits=max(max_abs_bits,abs(cost-float(actual['bits'])))
        max_abs_bpc=max(max_abs_bpc,abs(cost/source['chars']-float(actual['bpc'])))
        assert math.isclose(cost,float(actual['bits']),rel_tol=1e-12,abs_tol=2e-9),(name,source['filename'],source['view'],cost,actual['bits'])
    assert unknowns==0
    return dict(model=name,LM_tokens=n,score_rows=len(records),model_sha256=digest(mp),
                checked_score_sha256=digest(csvpath),max_abs_bits=max_abs_bits,
                max_abs_bpc=max_abs_bpc,unknown_tokens=unknowns)

def check_bootstrap(root,out):
    # Reconstruct independently, summing page costs by issue directly rather than trusting previous contrasts.
    df=pd.read_csv(out/'heldout_lm_scores.csv')
    assert len(df)==99720
    saved={(r['family'],r['period'],r['view'],r['contrast']):r for r in readcsv(out/'bpc_factorial_contrasts.csv')}
    assert len(saved)==32
    checks=[]
    for period in PERIODS:
        sub=df[df.period==period]
        issues=sorted(sub.issue.unique());num=len(issues)
        base_rng=np.random.default_rng(343401+(period!='1888'))
        samples=base_rng.multinomial(num,[1.0/num]*num,size=5000)
        if period=='1889-94':
            yrng=np.random.default_rng(343501)
            yrs=sorted(sub.year.unique())
            year_draw=yrng.multinomial(6,[1/6]*6,size=5000)
        for fam in FAM:
            for view in VIEWS:
                t=sub[(sub.family==fam)&(sub.view==view)]
                charmap=t[(t.arm=='PP')&(t.seed==441)].groupby('issue').chars.sum().to_dict()
                nc=np.array([charmap.get(issue,0) for issue in issues],dtype=float)
                assert nc.sum()>0
                grouped=t.groupby(['issue','arm','seed'],sort=False).bits.sum()
                numerator=np.array([[[grouped.get((issue,arm,seed),0.) for issue in issues] for seed in SEEDS] for arm in ARMS],dtype=float)
                pt=numerator.sum(axis=2)/nc.sum()
                boot_numer=np.einsum('bi,asi->bas',samples,numerator)
                boot_denom=samples@nc
                sample_rate=boot_numer/boot_denom[:,None,None]
                c_pt={a:pt[k] for k,a in enumerate(ARMS)}
                c_boot={a:sample_rate[:,k,:] for k,a in enumerate(ARMS)}
                def cs(a):
                    return {'allocation':.5*((a['WP']-a['PP'])+(a['WW']-a['PW'])),
                            'within_view':.5*((a['PW']-a['PP'])+(a['WW']-a['WP'])),
                            'interaction':(a['WW']-a['WP'])-(a['PW']-a['PP']),
                            'WW_minus_PP':a['WW']-a['PP']}
                points=cs(c_pt)
                draws=cs(c_boot)
                if period=='1889-94':
                    yearn=np.zeros((4,5,6));yearc=np.zeros(6)
                    for iy,year in enumerate(yrs):
                        idx=[i for i,issue in enumerate(issues) if int(issue[5:9])==year]
                        yearn[:,:,iy]=numerator[:,:,idx].sum(axis=2)
                        yearc[iy]=nc[idx].sum()
                    yrnum=np.einsum('by,asy->bas',year_draw,yearn)
                    yrden=year_draw@yearc
                    yrsamps=cs({a:yrnum[:,k,:]/yrden[:,None] for k,a in enumerate(ARMS)})
                for c,value in points.items():
                    q=saved[(fam,period,view,c)]
                    lo,hi=np.quantile(draws[c].mean(axis=1),[.025,.975])
                    diff=max(abs(value.mean()-float(q['mean'])),abs(lo-float(q['ci_low'])),abs(hi-float(q['ci_high'])))
                    if period=='1889-94':
                        yl,yh=np.quantile(yrsamps[c].mean(axis=1),[.025,.975])
                        diff=max(diff,abs(yl-float(q['year_cluster_ci_low'])),abs(yh-float(q['year_cluster_ci_high'])))
                    assert diff<2e-10,(fam,period,view,c,diff)
                    checks.append(dict(family=fam,period=period,view=view,contrast=c,max_abs_residual=diff))
    return checks

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--archive',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--max-new',type=int,default=0);args=ap.parse_args()
    root=Path(args.root);out=Path(args.output)
    assert digest(args.archive)=='375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5'
    assert digest(root/'Phase1'/'FROZEN_PROTOCOL.md')=='bc443d883fa1c2b8eb09ced5f87d8f65b352d55512ef2c594b1b23e64d496bf4'
    assert digest(root/'Phase5'/'PRECOMMIT.md')=='8575dc1ae52a22391318933bc5d4c8cee8777c15645a9eed13bbe34bb15583f8'
    freeze=json.loads((root/'Phase2'/'outputs'/'PHASE2_SELECTION_FREEZE.json').read_text())
    assert freeze['selected_alpha']==100 and freeze['heldout_content_reads']==0
    docs,pages=get_source_texts(args.archive)
    refdocs=readcsv(root/'Phase2'/'outputs'/'lm_training_manifest.csv')
    refeval=readcsv(root/'Phase3'/'outputs'/'heldout_pageview_manifest.csv')
    for a,b in zip(docs,refdocs):
        for k in ('filename','issue','view','text_sha256','chars'):assert str(a[k])==str(b[k]),('training',k,a,b)
    for a,b in zip(pages,refeval):
        for k in ('filename','view','issue','period','text_sha256','chars'):assert str(a[k])==str(b[k]),('eval',k,a,b)
    cache=out/'independent_verification_models';cache.mkdir(exist_ok=True)
    new=0
    for fam in FAM:
        for arm in ARMS:
            for seed in SEEDS:
                name=f'{fam}_{arm}_{seed}'
                dst=cache/f'{name}.json'
                if dst.exists():
                    x=json.loads(dst.read_text())
                    assert x['checked_score_sha256']==digest(out/'model_rows'/f'{name}.csv')
                    print('CHECKPOINT_OK',name,flush=True)
                    continue
                x=run_one(fam,arm,seed,docs,pages,root,out)
                dst.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
                new+=1
                print('INDEPENDENT_FORMULA_OK',name,'rows',x['score_rows'],'maxerr',format(x['max_abs_bits'],'.3g'),flush=True)
                if args.max_new and new>=args.max_new:
                    print('CHECKPOINT_ONLY',new,flush=True)
                    return
    results=[json.loads((cache/f'{fam}_{arm}_{seed}.json').read_text()) for fam in FAM for arm in ARMS for seed in SEEDS]
    assert len(results)==40 and sum(v['score_rows'] for v in results)==99720
    assert max(r['max_abs_bits'] for r in results)<2e-9
    bootstrap_checks=check_bootstrap(root,out)
    assert len(bootstrap_checks)==32
    dec=json.loads((out/'F5_DECISION.json').read_text())
    assert dec['F5']=='SUPPORTED' and dec['alpha']==100
    res={'pass':True,'independent_source_extraction':True,'independent_bigram_formula':True,
         'raw_LM_documents_checked':len(docs),'raw_heldout_pageviews_checked':len(pages),
         'distinct_models':len(results),'model_scores_checked':sum(r['score_rows'] for r in results),
         'independent_bootstrap_rows':len(bootstrap_checks),
         'max_abs_bigram_bits_error':max(r['max_abs_bits'] for r in results),
         'max_abs_bpc_error':max(r['max_abs_bpc'] for r in results),
         'max_abs_bootstrap_difference':max(x['max_abs_residual'] for x in bootstrap_checks),
         'zero_unknown_tokens':all(r['unknown_tokens']==0 for r in results),
         'score_files_sha256':[{'model':r['model'],'sha256':r['checked_score_sha256']} for r in results],
         'phase5_precommit_sha256':digest(root/'Phase5'/'PRECOMMIT.md'),
         'phase2_selection_sha256':digest(root/'Phase2'/'outputs'/'PHASE2_SELECTION_FREEZE.json')}
    (out/'PHASE5_INDEPENDENT_VERIFICATION.json').write_text(json.dumps(res,indent=2,sort_keys=True)+'\n')
    print('INDEPENDENT_ALL_PASS',json.dumps({k:v for k,v in res.items() if k!='score_files_sha256'},indent=2),flush=True)

if __name__=='__main__':main()
