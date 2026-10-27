#!/usr/bin/env python3
"""Prospective 2026-10-24 Phase 32: historical archive-assigned language visibility.

No semantic language labels are inferred. Usage: python run_phase32.py --zip ... --out ...
"""
import argparse,csv,hashlib,io,json,re,zipfile
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

PAT=re.compile(r'^(IMVO|ISIG)_(\d{4})-(\d{2})-(\d{2})_p(\d{3})\.txt$')

def quant(x):
    return [float(np.quantile(x,p)) for p in (.025,.975)]

def boot_issue_page_vs_word(df,B=3000,seed=3202):
    rng=np.random.default_rng(seed)
    grp=[g[['en','total','share']].to_numpy() for _,g in df.groupby(['source','date'],sort=True)]
    m=len(grp)
    mat=np.array([[a[:,0].sum(), a[:,1].sum(),a[:,2].sum(),len(a)] for a in grp],float)
    idx=rng.integers(0,m,size=(B,m))
    s=mat[idx].sum(axis=1)
    dif=s[:,0]/s[:,1]-s[:,2]/s[:,3]
    return {'interval':quant(dif),'replicate_mean':float(dif.mean()),'n_issues':m}

def bootstrap_mean_by_issue(arrA, arrB, seed=3203,B=3000):
    rng=np.random.default_rng(seed)
    a=np.asarray(arrA);b=np.asarray(arrB)
    ia=rng.integers(0,len(a),size=(B,len(a)))
    ib=rng.integers(0,len(b),size=(B,len(b)))
    ds=b[ib].mean(axis=1)-a[ia].mean(axis=1)
    return {'delta_late_minus_early':float(b.mean()-a.mean()),'ci':quant(ds)}

def source_month_bootstrap(issue_df, B=3000, seed=3201):
    x=issue_df[issue_df.year.isin([1885,1887,1888])].copy()
    d={}
    for (ym,source),sub in x.groupby(['year_month','source']):
        d[(ym,source)]=sub.share.to_numpy()
    cells=sorted(set(ym for ym,s in d if (ym,'IMVO') in d and (ym,'ISIG') in d))
    observed=float(np.mean([d[(ym,'IMVO')].mean()-d[(ym,'ISIG')].mean() for ym in cells]))
    rng=np.random.default_rng(seed)
    diffs=np.zeros(B)
    for ym in cells:
        a=d[(ym,'IMVO')];b=d[(ym,'ISIG')]
        ia=rng.integers(0,len(a),size=(B,len(a)))
        ib=rng.integers(0,len(b),size=(B,len(b)))
        diffs+=(a[ia].mean(axis=1)-b[ib].mean(axis=1))/len(cells)
    # cell bootstrap: naturally conditions on a sample of shared year-month slots, not fixed cells.
    cd=np.array([d[(ym,'IMVO')].mean()-d[(ym,'ISIG')].mean() for ym in cells])
    ic=rng.integers(0,len(cd),size=(B,len(cd)))
    cbs=cd[ic].mean(axis=1)
    return {'observed_difference':observed,'issue_ci':quant(diffs),'shared_year_month_cells':len(cells),
            'cell_bootstrap_ci_sensitivity':quant(cbs),'cell_differences':dict(zip(cells,cd.tolist()))}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--zip',required=True)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
    contents={}
    with zipfile.ZipFile(args.zip) as z:
        names=z.namelist()
        for field in ('XhTxt','EnTxt'):
            for path in names:
                if f'/{field}/' in path and path.endswith('.txt') and '/._' not in path:
                    basename=path.rsplit('/',1)[-1]
                    if not PAT.fullmatch(basename):continue
                    contents[(basename,field)]=len(z.read(path).decode('utf-8-sig',errors='replace').split())
        tname=next(n for n in names if n.endswith('LanguageSplitTracker70-94.csv'))
        tracker=pd.read_csv(io.BytesIO(z.read(tname)))
    rows=[]
    for _,tr in tracker.iterrows():
        fn=str(tr['Edition'])
        mt=PAT.match(fn)
        if not mt:continue
        src,yr,mo,da,pageno=mt.groups()
        xh=contents.get((fn,'XhTxt'))
        en=contents.get((fn,'EnTxt'))
        if xh is None or en is None:continue
        total=xh+en
        raw=int(tr['Tot_Wordcount'])
        rows.append({'filename':fn,'source':src,'date':f'{yr}-{mo}-{da}',
             'year':int(yr),'month':int(mo),'year_month':f'{yr}-{mo}',
             'page':int(pageno),'xh':xh,'en':en,'total':total,
             'share':en/total if total else np.nan,
             'raw_tracker_words':raw,'unclassified':raw-total,
             'xh_tracker':int(tr['Xho_Wordcount']),'en_tracker':int(tr['Eng_Wordcount'])})
    df=pd.DataFrame(rows)
    if len(df)!=3608: raise RuntimeError(f'Expected 3608 aligned pages, got {len(df)}')
    if df['share'].isna().any():
        print('Dropping zero-denominator pages',df['share'].isna().sum())
    valid=df[df.total>0].copy()
    df.to_csv(out/'page_visibility.csv',index=False)
    issue=(valid.groupby(['source','date','year','month','year_month'],as_index=False)
           .agg(xh=('xh','sum'),en=('en','sum'),total=('total','sum'),pages=('filename','count')))
    issue['share']=issue['en']/issue['total']
    issue.to_csv(out/'issue_visibility.csv',index=False)
    yearly=(valid.groupby(['source','year'],as_index=False).agg(
             en=('en','sum'),xh=('xh','sum'),words=('total','sum'),pages=('filename','count'),issues=('date','nunique'),page_mean_share=('share','mean')))
    yearly['word_share']=yearly.en/yearly.words
    yearly=yearly.merge(issue.groupby(['source','year'],as_index=False).agg(issue_mean_share=('share','mean')),on=['source','year'])
    yearly.to_csv(out/'yearly_visibility.csv',index=False)
    # sampling effects (word, page, issue) per source and period
    periods=[('overlap_1885_1887_1888',[1885,1887,1888]),('imvo_early_1885_1888',list(range(1885,1889))),
         ('imvo_late_1889_1894',list(range(1889,1895))),('all_years',list(range(1870,1895)))]
    effects=[]
    for source in ('IMVO','ISIG'):
        for pname,ys in periods:
            q=valid[(valid.source==source)&(valid.year.isin(ys))]
            if len(q)==0:continue
            qi=issue[(issue.source==source)&(issue.year.isin(ys))]
            effects.append({'source':source,'period':pname,'pages':len(q),'issues':len(qi),
              'word_weighted':q.en.sum()/q.total.sum(),'page_weighted':q.share.mean(),
              'issue_weighted':qi.share.mean(),'word_minus_page':q.en.sum()/q.total.sum()-q.share.mean(),
              'word_minus_issue':q.en.sum()/q.total.sum()-qi.share.mean()})
    eff=pd.DataFrame(effects);eff.to_csv(out/'sampling_estimands.csv',index=False)
    # H32a
    h32a=source_month_bootstrap(issue)
    # H32b
    q=valid[(valid.source=='IMVO')&(valid.year.isin([1885,1887,1888]))]
    h32b=boot_issue_page_vs_word(q)
    h32b.update({'observed_difference':float(q.en.sum()/q.total.sum()-q.share.mean())})
    # H32c
    early=issue[(issue.source=='IMVO')&(issue.year.between(1885,1888))]
    late=issue[(issue.source=='IMVO')&(issue.year.between(1889,1894))]
    h32c=bootstrap_mean_by_issue(early.share,late.share)
    # year equal sensitivity + year-cluster bootstrap
    y_early=yearly[(yearly.source=='IMVO')&(yearly.year.between(1885,1888))].issue_mean_share.to_numpy()
    y_late=yearly[(yearly.source=='IMVO')&(yearly.year.between(1889,1894))].issue_mean_share.to_numpy()
    h32c['year_equal_delta']=float(y_late.mean()-y_early.mean())
    h32c['year_cluster_bootstrap']=bootstrap_mean_by_issue(y_early,y_late,seed=3204)['ci']
    h32c['early_issue_count']=len(early);h32c['late_issue_count']=len(late)
    # concentration
    conc=[]
    for source in ('IMVO','ISIG'):
        q=valid[valid.source==source]
        sorted_en=q.sort_values('en',ascending=False).en.to_numpy()
        topdec=max(1,int(np.ceil(len(sorted_en)*.1)))
        conc.append({'source':source,'pages':len(q),'english_words':int(q.en.sum()),
                     'top_10pct_page_share_of_all_english':float(sorted_en[:topdec].sum()/sorted_en.sum())})
    # Page location
    pos=valid.copy()
    pos['page_bin']=pos.page.clip(upper=4)
    page_pos=(pos.groupby(['source','page_bin'],as_index=False).agg(en=('en','sum'),total=('total','sum'),n_pages=('filename','count')))
    page_pos['en_word_share']=page_pos.en/page_pos.total
    page_pos.to_csv(out/'page_position_visibility.csv',index=False)
    # Integrity
    mismatch=valid[(valid.xh!=valid.xh_tracker)|(valid.en!=valid.en_tracker)]
    integrity={'num_pages_tracker':len(tracker),'num_aligned_pages':len(df),'num_nonempty_pages':len(valid),
      'mismatch_page_count':len(mismatch),'page_mismatches_first_five':mismatch.filename.head().tolist(),
      'classified_words_total':int(df.total.sum()),'unclassified_words_total':int(df.unclassified.sum()),
      'unclassified_words_rate':float(df.unclassified.sum()/df.raw_tracker_words.sum()),
      'negative_unclassified_pages':int((df.unclassified<0).sum()),
      'blank_pages':int((df.total==0).sum())}
    result={'h32a':h32a,'h32b':h32b,'h32c':h32c,'h32d':conc,'integrity':integrity}
    # Decision gates exactly as preregistered
    result['decisions']={
      'H32a_supported':bool(h32a['observed_difference']>=.10 and h32a['issue_ci'][0]>0),
      'H32b_supported':bool(h32b['observed_difference']>.05 and h32b['interval'][0]>0),
      'H32c_supported':bool(abs(h32c['delta_late_minus_early'])>=.05 and
                           (h32c['ci'][0]>0 or h32c['ci'][1]<0))}
    (out/'RESULTS.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
