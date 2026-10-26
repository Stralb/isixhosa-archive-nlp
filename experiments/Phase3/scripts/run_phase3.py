#!/usr/bin/env python3
"""Excellence Sprint Phase 3 frozen held-out factorial evaluation.

Scores ONLY the 1888-1894 IMVO EnTxt/XhTxt pages, with the identical
frozen Phase 2 tokenizers; no refits, model selection, or test tuning.
Clustering: publication issue (year-month-day), 5000 paired draws.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, re, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import sentencepiece as spm

EXPECTED_ARCHIVE = '375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5'
EXPECTED_PROTOCOL = 'bc443d883fa1c2b8eb09ced5f87d8f65b352d55512ef2c594b1b23e64d496bf4'
WORDS = re.compile(r'[a-z]+')
FILE = re.compile(r'IMVO_(\d{4})-(\d{2})-(\d{2})_p\d{3}\.txt$')
FAMILIES = ('bpe', 'unigram')
ARMS = ('PP','PW','WP','WW')
SEEDS = (441,442,443,444,445)
VIEWS = ('en','xh')
PERIODS = ('1888','1889-94')
N_BOOT = 5000
BOOT_SEED = 343401


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while True:
            x=f.read(2*1024*1024)
            if not x:break
            h.update(x)
    return h.hexdigest()


def csvread(path):
    with open(path, newline='', encoding='utf-8') as f: return list(csv.DictReader(f))


def writecsv(path,rows,fields):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def load_heldout(archive):
    pages={}
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():
            path=member.split('/')
            if len(path)<2 or path[-2] not in ('EnTxt','XhTxt'):continue
            name=path[-1];m=FILE.fullmatch(name)
            if not m:continue
            year,month,day=map(int,m.groups())
            if not 1888<=year<=1894:continue
            view='en' if path[-2]=='EnTxt' else 'xh'
            key=(name,view)
            assert key not in pages,('duplicate archive key',key)
            words=WORDS.findall(z.read(member).decode('utf-8-sig',errors='replace').lower())
            # Phase 1 frozen evaluation: first 128 alpha words, 40-word gate per view.
            if len(words)<40:continue
            segment=words[:128];text=' '.join(segment)
            assert 40<=len(segment)<=128
            chars=sum(len(x) for x in segment)
            assert chars>0
            pages[key]=dict(filename=name,view=view,year=year,issue=f'IMVO_{year:04d}-{month:02d}-{day:02d}',
                    period='1888' if year==1888 else '1889-94',words=len(segment),total_view_words=len(words),
                    chars=chars,text_sha256=hashlib.sha256(text.encode()).hexdigest(),text=text)
    assert pages and len({x['issue'] for x in pages.values()})>300
    return [pages[k] for k in sorted(pages)]


def audit_models(root):
    manifest=csvread(root/'Phase2'/'outputs'/'model_training_manifest.csv')
    expected={f'{f}_{a}_{s}' for f in FAMILIES for a in ARMS for s in SEEDS}
    assert len(manifest)==40 and {r['model_name'] for r in manifest}==expected
    models={}
    for rec in manifest:
        name=rec['model_name'];model=root/'Phase2'/'outputs'/'models'/f'{name}.model'
        assert digest(model)==rec['model_sha256'],name
        assert digest(model.with_suffix('.vocab'))==rec['vocab_sha256'],name
        assert int(rec['validation_unknown_tokens'])==0
        assert 499970<=int(rec['actual_alpha_chars'])<=500000
        assert int(rec['model_vocab_size'])==2000
        models[name]={'path':model,'source':rec}
    choice=json.loads((root/'Phase2'/'outputs'/'PHASE2_SELECTION_FREEZE.json').read_text())
    assert choice['selected_alpha']==100 and choice['models_count']==40
    assert choice['phase1_protocol_sha256']==EXPECTED_PROTOCOL
    assert choice['archive_sha256']==EXPECTED_ARCHIVE
    return models,choice


def score_pages(pages,models):
    result=[]
    for i,(name,data) in enumerate(sorted(models.items())):
        family,arm,seed=name.split('_');seed=int(seed)
        sp=spm.SentencePieceProcessor(model_file=str(data['path']))
        assert sp.vocab_size()==2000
        for r in pages:
            ids=sp.encode(r['text'],out_type=int)
            result.append(dict(family=family,arm=arm,seed=seed,filename=r['filename'],view=r['view'],
                  issue=r['issue'],year=r['year'],period=r['period'],chars=r['chars'],words=r['words'],
                  text_sha256=r['text_sha256'],pieces=len(ids),unknown=sum(j==sp.unk_id() for j in ids)))
        print('SCORED',i+1,'/',len(models),name, 'page_views',len(pages),flush=True)
    return result


def arm_metrics(t):
    # Each family/period/view/seed/arm gets a corpus-char-weighted metric.
    group=t.groupby(['family','period','view','seed','arm'],sort=True).agg(
            pieces=('pieces','sum'),chars=('chars','sum'),unknown=('unknown','sum'),
            page_views=('filename','size'),issues=('issue','nunique')).reset_index()
    group['pieces_per_100_chars']=100*group['pieces']/group['chars']
    group['unknown_rate_per_100_pieces']=100*group['unknown']/group['pieces']
    return group


def contrasts(metric_dict):
    a=metric_dict
    # Positive means more pieces (worse), negative means fewer pieces (better).
    da=.5*((a['WP']-a['PP'])+(a['WW']-a['PW']))
    db=.5*((a['PW']-a['PP'])+(a['WW']-a['WP']))
    di=(a['WW']-a['WP'])-(a['PW']-a['PP'])
    total=a['WW']-a['PP']
    return {'allocation':da,'within_view':db,'interaction':di,'WW_minus_PP':total}


def main_bootstrap(t):
    # Use same issue draws, for ALL four arms and both model families and views.
    out=[];issue_meta=[];year_rows=[]
    for period in PERIODS:
        q=t[t['period']==period]
        issues=sorted(q['issue'].unique());n=len(issues)
        assert n>10, (period,n)
        # Precommitted identical issue random generator, sequential per period.
        rng=np.random.default_rng(BOOT_SEED + (0 if period=='1888' else 1))
        wt=rng.multinomial(n,[1./n]*n,size=N_BOOT).astype(float)
        if period=='1889-94':
            years=sorted(q['year'].unique());assert years==list(range(1889,1895))
            rngyear=np.random.default_rng(BOOT_SEED+100)
            yearwt=rngyear.multinomial(len(years),[1./len(years)]*len(years),size=N_BOOT).astype(float)
        else:yearwt=None;years=None
        issue_meta.append({'period':period,'issues':n,'bootstrap_draws':N_BOOT,'issue_seed':BOOT_SEED+(0 if period=='1888' else 1),
                          'six_year_cluster_seed':BOOT_SEED+100 if period=='1889-94' else ''})
        for fam in FAMILIES:
            for v in VIEWS:
                df=q[(q['family']==fam)&(q['view']==v)]
                # Every model sees identical pages, and all 40 expected cells exist.
                expected={(a,s) for a in ARMS for s in SEEDS}
                assert set(zip(df.arm,df.seed))==expected
                # Per-issue char budgets are fixed across arms and seeds.
                char_by_issue=df[(df.arm=='PP')&(df.seed==SEEDS[0])].groupby('issue')['chars'].sum()
                chars=np.array([int(char_by_issue.get(issue,0)) for issue in issues],dtype=float)
                assert chars.sum()>0
                indexed=df.groupby(['arm','seed','issue'])['pieces'].sum()
                piece=np.zeros((len(ARMS),len(SEEDS),n),dtype=float)
                for i,a in enumerate(ARMS):
                    for j,s in enumerate(SEEDS):
                        piece[i,j]=[float(indexed.get((a,s,issue),0)) for issue in issues]
                total=chars.sum()
                point=100*piece.sum(axis=2)/total  # arm x seed
                denom=wt @ chars
                valid=denom>0
                assert np.mean(valid)>.999, (period,v,np.mean(valid))
                # nboot x arm x seeds, weight entire publication issues.
                sampled=(wt @ piece.reshape(len(ARMS)*len(SEEDS),n).T).reshape(N_BOOT,len(ARMS),len(SEEDS))
                samples=100*sampled/denom[:,None,None]
                m={a:point[i] for i,a in enumerate(ARMS)}
                d=contrasts(m)
                boots=contrasts({a:samples[:,i,:] for i,a in enumerate(ARMS)})
                for contrast in d:
                    seed_vals=np.asarray(d[contrast]);mean=float(seed_vals.mean())
                    bootmean=np.asarray(boots[contrast]).mean(axis=1)
                    lo,hi=np.quantile(bootmean,[.025,.975])
                    entry=dict(family=fam,period=period,view=v,contrast=contrast,mean=mean,ci_low=float(lo),ci_high=float(hi),
                         seeds=len(SEEDS),issues=n,eligible_page_views=len(df)//(len(ARMS)*len(SEEDS)),
                         seed_min=float(seed_vals.min()),seed_max=float(seed_vals.max()),
                         seeds_negative=int(sum(seed_vals<0)),seeds_positive=int(sum(seed_vals>0)))
                    out.append(entry)
                    for s,sv in zip(SEEDS,seed_vals):
                        year_rows.append(dict(family=fam,period=period,view=v,seed=s,year='all',contrast=contrast,value=float(sv)))
                if yearwt is not None:
                    byyear={y:i for i,y in enumerate(years)}
                    char_year=np.array([sum(chars[k] for k,issue in enumerate(issues) if int(issue[5:9])==y) for y in years])
                    piece_year=np.zeros((len(ARMS),len(SEEDS),len(years)))
                    for k,issue in enumerate(issues):piece_year[:,:,byyear[int(issue[5:9])]]+=piece[:,:,k]
                    denom_yr=yearwt @ char_year
                    assert (denom_yr>0).all()
                    sample_yr=(yearwt@piece_year.reshape(len(ARMS)*len(SEEDS),len(years)).T).reshape(N_BOOT,len(ARMS),len(SEEDS))
                    contrasts_yr=contrasts({a:100*sample_yr[:,i,:]/denom_yr[:,None] for i,a in enumerate(ARMS)})
                    for contrast,bs in contrasts_yr.items():
                        lo,hi=np.quantile(np.asarray(bs).mean(axis=1),[.025,.975])
                        for row in out:
                            if row['family']==fam and row['period']==period and row['view']==v and row['contrast']==contrast:
                                row['year_cluster_ci_low']=float(lo);row['year_cluster_ci_high']=float(hi)
                    for j,y in enumerate(years):
                        x=contrasts({a:100*piece_year[i,:,j]/char_year[j] for i,a in enumerate(ARMS)})
                        for c,ss in x.items():
                            year_rows.append(dict(family=fam,period=period,view=v,seed='mean',year=y,contrast=c,value=float(np.mean(ss))))
        print('BOOTSTRAPPED',period,'publication_issues',n,'draws',N_BOOT,flush=True)
    return out,issue_meta,year_rows


def make_decisions(contr):
    lookup={(r['family'],r['period'],r['view'],r['contrast']):r for r in contr}
    def g(f,p,v,c):return lookup[(f,p,v,c)]
    f1=g('bpe','1888','en','allocation');late1=g('bpe','1889-94','en','allocation')
    f2=g('unigram','1888','en','allocation');late2=g('unigram','1889-94','en','allocation')
    f3=g('bpe','1888','xh','allocation');late3=g('bpe','1889-94','xh','allocation')
    f4=g('unigram','1888','en','within_view');late4=g('unigram','1889-94','en','within_view')
    r=[('F1','BPE English view allocation effect',f1,f1['mean']<=-.30 and f1['ci_high']<0 and late1['mean']<0,late1),
       ('F2','Unigram English view allocation effect',f2,f2['mean']<=-.30 and f2['ci_high']<0 and late2['mean']<0,late2),
       ('F3','BPE isiXhosa view tradeoff',f3,f3['mean']>0 and f3['ci_low']>0 and late3['mean']>0,late3),
       ('F4','Unigram English view within-view effect',f4,abs(f4['mean'])>=.30 and not(f4['ci_low']<=0<=f4['ci_high']) and (late4['mean']*f4['mean']>0),late4)]
    return [dict(hypothesis=h,description=descr,status=('SUPPORTED' if ok else 'NOT_SUPPORTED'),
                 effect_1888=x['mean'],ci_1888_low=x['ci_low'],ci_1888_high=x['ci_high'],
                 effect_1889_94=y['mean'],ci_1889_94_low=y['ci_low'],ci_1889_94_high=y['ci_high'],
                 criterion_met=bool(ok)) for h,descr,x,ok,y in r]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--archive',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();root=Path(args.root);archive=Path(args.archive);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    assert digest(archive)==EXPECTED_ARCHIVE
    assert digest(root/'Phase1'/'FROZEN_PROTOCOL.md')==EXPECTED_PROTOCOL
    models,freeze=audit_models(root)
    pages=load_heldout(archive)
    manifest=[{k:v for k,v in r.items() if k!='text'} for r in pages]
    writecsv(out/'heldout_pageview_manifest.csv',manifest,list(manifest[0].keys()))
    data=score_pages(pages,models)
    writecsv(out/'heldout_scores.csv',data,list(data[0].keys()))
    df=pd.DataFrame(data)
    byarm=arm_metrics(df)
    byarm.to_csv(out/'per_arm_seed_metrics.csv',index=False,float_format='%.14g')
    contrast,meta,yearrows=main_bootstrap(df)
    writecsv(out/'factorial_contrasts.csv',contrast,list(dict.fromkeys(k for row in contrast for k in row)))
    writecsv(out/'bootstrap_design.csv',meta,list(meta[0].keys()))
    writecsv(out/'per_seed_and_year_effects.csv',yearrows,list(yearrows[0].keys()))
    hyp=make_decisions(contrast)
    writecsv(out/'hypothesis_decisions.csv',hyp,list(hyp[0].keys()))
    result={'PHASE3_TEST_EVALUATED':True,'phase1_protocol_sha256':digest(root/'Phase1'/'FROZEN_PROTOCOL.md'),
       'phase2_selection_sha256':digest(root/'Phase2'/'outputs'/'PHASE2_SELECTION_FREEZE.json'),
       'source_archive_sha256':digest(archive),'fixed_alpha':freeze['selected_alpha'],
       'models_verified':len(models),'eligible_page_views':len(pages),'model_scores':len(data),
       'periods':{p:{'page_views':sum(x['period']==p for x in pages),'issues':len({x['issue'] for x in pages if x['period']==p}),
          'en_page_views':sum(x['period']==p and x['view']=='en' for x in pages),
          'xh_page_views':sum(x['period']==p and x['view']=='xh' for x in pages)} for p in PERIODS},
       'unknown_tokens':int(df.unknown.sum()),'decisions':hyp,
       'analyst_level_reuse_warning':'1888-1894 corpus had been studied in earlier phases; conditional test, not untouched independent validation',
       'phase4_reprint_control_ran':False,'phase5_lm_test_ran':False}
    (out/'phase3_manifest.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print('RESULT_MANIFEST',json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
