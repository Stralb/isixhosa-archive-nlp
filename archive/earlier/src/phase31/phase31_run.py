#!/usr/bin/env python3
"""Phase31 out-of-time BPE token-budget experiment. See precommits before interpreting."""
from __future__ import annotations
import argparse, collections, csv, hashlib, io, json, re, statistics, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import sentencepiece as spm

BASE=Path(__file__).resolve().parent
DATA_DEFAULT='/mnt/data/TxtLangFiles1870-1894_TB.zip'
MODELS_DEFAULT='/mnt/data/isiXhosa_Phase30_Shared_Word_Frequency_BPE_2026-10-24.zip'
PAT=re.compile(r'(IMVO|ISIG)_(\d{4})-(\d{2})-(\d{2})_p(\d+)\.txt$')
WORD=re.compile(r'[a-z]+')
BUDGETS=[192,224,256,288,320]
SEEDS=range(3)
M_NAMES=['IMVO','ISIG','BAL_A','BAL_B']

def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def get_windows(path:Path):
    windows=[]; excluded=collections.Counter(); all_files=collections.Counter()
    with zipfile.ZipFile(path) as z:
        names=sorted(n for n in z.namelist() if '/XhTxt/' in n and n.endswith('.txt') and '/._' not in n)
        for name in names:
            m=PAT.search(name)
            if not m:excluded['invalid_filename']+=1;continue
            source,year,month,day,page=m.groups();year=int(year)
            if source=='ISIG' and year<=1884:group='EARLY_ISIG'
            elif source=='IMVO' and year>=1889:group='LATE_IMVO'
            else:continue
            all_files[group]+=1
            # No decoding fallback is needed for this archive, but errors are explicitly replaced, logged in audits.
            raw=z.read(name).decode('utf-8-sig',errors='replace').replace('\r\n','\n')
            paragraphs=re.split(r'\n\s*\n+',raw)
            tokenized=[WORD.findall(p.lower()) for p in paragraphs]
            eligible=[(len(toks),-i,i,toks) for i,toks in enumerate(tokenized)]
            if not eligible or max(x[0] for x in eligible)<90:
                excluded[f'{group}_short_page']+=1;continue
            _,_,idx,toks=max(eligible,key=lambda it:(it[0],it[1]))
            words=toks[:90];assert len(words)==90
            alpha=sum(map(len,words));assert alpha>0
            file=Path(name).name
            windows.append({'file':file,'source':source,'group':group,'year':year,'month':int(month),'issue':f'{source}_{year:04d}-{month}-{day}',
                            'paragraph_index':idx,'words':words,'alpha_chars':alpha,'window_sha256':sha(' '.join(words).encode())})
    assert len(set(w['file'] for w in windows))==len(windows)
    assert all(w['year']<=1884 or w['year']>=1889 for w in windows)
    return windows,excluded,all_files

def models_from_zip(path:Path):
    z=zipfile.ZipFile(path)
    byname={}
    hashes={}
    for seed in SEEDS:
        for typ in M_NAMES:
            if typ=='IMVO':name=f'Phase30/inputs/BPE_IMVO_seed{seed}.model'
            elif typ=='ISIG':name=f'Phase30/inputs/BPE_ISIG_seed{seed}.model'
            elif typ=='BAL_A':name=f'Phase30/LINE_PLACEBO_A_seed{seed}.model'
            else:name=f'Phase30/LINE_PLACEBO_B_seed{seed}.model'
            b=z.read(name)
            hashes[f'{typ}_{seed}']={'original_zip_member':name,'sha256':sha(b),'bytes':len(b)}
            proc=spm.SentencePieceProcessor(model_proto=b)
            if proc.get_piece_size()<100:raise RuntimeError('unexpectedly tiny model')
            byname[typ,seed]=proc
    z.close()
    return byname,hashes

def writecsv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf8') as fh:
        out=csv.DictWriter(fh,fieldnames=list(rows[0].keys()));out.writeheader();out.writerows(rows)

def score(windows,models):
    # Each tokenizer sees exact same 90-form lowercased text.
    rows=[];mismatches=[]
    word_cache=collections.defaultdict(dict)
    for ix,w in enumerate(windows):
        full=' '.join(w['words'])
        for seed in SEEDS:
            for typ in M_NAMES:
                sp=models[typ,seed]
                ids=sp.encode(full,out_type=int);total=len(ids)
                cs=word_cache[typ,seed]
                wp=[]
                for t in w['words']:
                    if t not in cs:cs[t]=len(sp.encode(t,out_type=int))
                    wp.append(cs[t])
                additive=sum(wp)
                if total!=additive:
                    mismatches.append((w['file'],typ,seed,total,additive))
                    # Use exact prefix length for retention if additivity fails, at modest additional cost.
                    sums=[]
                    for i in range(1,91):sums.append(len(sp.encode(' '.join(w['words'][:i]),out_type=int)))
                else:
                    sums=np.cumsum(wp)
                n_unk=int(sum(x==sp.unk_id() for x in ids))
                vals=dict(file=w['file'],source=w['source'],group=w['group'],issue=w['issue'],year=w['year'],month=w['month'],
                          alpha_chars=w['alpha_chars'],seed=seed,model=typ,pieces=total,E_pieces_per_100_alpha=100*total/w['alpha_chars'],
                          unknown_pieces=n_unk,n_words=90,full_fit_256=int(total<=256),window_sha256=w['window_sha256'])
                for b in BUDGETS: vals[f'retain_{b}']=int(np.searchsorted(sums,b,side='right'))
                rows.append(vals)
    return rows,mismatches

def wide_issues(df):
    # Average each page model's repeated seed scores, then issues, including window counts.
    keys=['group','issue','year','model']
    valcols=['E_pieces_per_100_alpha','pieces','full_fit_256']+[f'retain_{b}' for b in BUDGETS]
    issue_model=df.groupby(keys,as_index=False)[valcols].mean()
    wide=issue_model.pivot(index=['group','issue','year'],columns='model',values=valcols)
    wide.columns=[f'{a}_{b}' for a,b in wide.columns];wide=wide.reset_index()
    assert wide.notna().all().all()
    for metric in valcols:
        wide[f'{metric}_BALANCED']=(wide[f'{metric}_BAL_A']+wide[f'{metric}_BAL_B'])/2
    for metric in ['E_pieces_per_100_alpha'] + [f'retain_{b}' for b in BUDGETS]:
        wide[f'{metric}_OWN']=np.where(wide['group']=='EARLY_ISIG',wide[f'{metric}_ISIG'],wide[f'{metric}_IMVO'])
        wide[f'{metric}_FOREIGN']=np.where(wide['group']=='EARLY_ISIG',wide[f'{metric}_IMVO'],wide[f'{metric}_ISIG'])
    wide['specialist_E_gain']=wide['E_pieces_per_100_alpha_FOREIGN']-wide['E_pieces_per_100_alpha_OWN']
    for b in BUDGETS:wide[f'specialist_retained_gain_{b}']=wide[f'retain_{b}_OWN']-wide[f'retain_{b}_FOREIGN']
    return wide

def h31c(dat):
    vals={g:dat[g] for g in ('EARLY_ISIG','LATE_IMVO')}
    # For each source, own specialist is ISIG in early / IMVO in late.
    gaps={}
    for model in ['IMVO','ISIG','BALANCED']:
        r=[]
        for g,v in vals.items():
            own=float(v['retain_256_OWN'].mean());alt=float(v[f'retain_256_{model}'].mean());r.append(max(0.,own-alt))
        gaps[model]=float(max(r))
    best_single=min(gaps['IMVO'],gaps['ISIG']);adv=best_single-gaps['BALANCED']
    frac=adv/best_single if best_single>1e-12 else None
    return {'best_exclusive_worst_regret':best_single,'balanced_worst_regret':gaps['BALANCED'],'gain_words':adv,'fraction_reduced':frac,
             'Imvo_exclusive_worst_regret':gaps['IMVO'],'Isig_exclusive_worst_regret':gaps['ISIG']}

def compute_summary(df,issues,nboot=2000):
    rng=np.random.default_rng(31024)
    rows=[];samples={};ci={};groups=['EARLY_ISIG','LATE_IMVO']
    for group in groups:
        g=issues.loc[issues.group==group].reset_index(drop=True)
        assert len(g)>10
        fields=['specialist_E_gain','specialist_retained_gain_256']
        mat=g[fields].to_numpy(float)
        ixs=rng.integers(0,len(g),size=(nboot,len(g)))
        boots=mat[ixs].mean(axis=1)
        arr=mat.mean(axis=0)
        samples[group]=g
        ci[group]={}
        for i,f in enumerate(fields):
            low,high=map(float,np.quantile(boots[:,i],[.025,.975]))
            ci[group][f]=(low,high)
            rows.append(dict(group=group,metric=f,point_estimate=float(arr[i]),ci_lower=low,ci_upper=high,n_issues=len(g),n_pages=int(df[df.group==group].file.nunique())))
    combined={g:samples[g] for g in groups}
    h=h31c(combined)
    boot_c=[]
    for i in range(nboot):
        subset={g:frame.iloc[rng.integers(0,len(frame),len(frame))] for g,frame in combined.items()}
        boot_c.append(h31c(subset)['gain_words'])
    h['gain_95ci']=list(map(float,np.quantile(boot_c,[.025,.975])))
    H31a=all(ci[g]['specialist_E_gain'][0]>0 for g in groups)
    H31b=all(float(samples[g]['specialist_retained_gain_256'].mean())>=2 and ci[g]['specialist_retained_gain_256'][0]>0 for g in groups)
    H31c=(h['fraction_reduced'] is not None and h['fraction_reduced']>=.5 and h['gain_95ci'][0]>0)
    decisions={'H31a_out_of_period_specialisation':bool(H31a),'H31b_at_least_two_extra_words_at_256':bool(H31b),
               'H31c_balanced_reduces_best_fixed_regret_at_least_half':bool(H31c)}
    metrics={}
    for group in groups:
        g=samples[group];metrics[group]={
            'issues':len(g),'pages':int(df.loc[df.group==group,'file'].nunique()),
            'specialist_E_gain_pieces_per_100_chars':float(g.specialist_E_gain.mean()),
            'specialist_E_gain_95CI':list(ci[group]['specialist_E_gain']),
            'specialist_retained_gain_words_B256':float(g.specialist_retained_gain_256.mean()),
            'specialist_retained_gain_95CI':list(ci[group]['specialist_retained_gain_256']),
            'B256_retained_OWN':float(g.retain_256_OWN.mean()),
            'B256_retained_FOREIGN':float(g.retain_256_FOREIGN.mean()),
            'B256_retained_BALANCED':float(g.retain_256_BALANCED.mean()),
            'B256_fullfit_rate_IMVO':float(df[(df.group==group)&(df.model=='IMVO')].full_fit_256.mean()),
            'B256_fullfit_rate_ISIG':float(df[(df.group==group)&(df.model=='ISIG')].full_fit_256.mean()),
            'B256_fullfit_rate_BALANCED':float(df[(df.group==group)&(df.model.isin(['BAL_A','BAL_B']))].full_fit_256.mean()),
            'secondary_budgets':{str(b):{'OWN':float(g[f'retain_{b}_OWN'].mean()),'FOREIGN':float(g[f'retain_{b}_FOREIGN'].mean()),'BALANCED':float(g[f'retain_{b}_BALANCED'].mean()),'specialist_gain':float(g[f'specialist_retained_gain_{b}'].mean())}for b in BUDGETS if b!=256}
        }
    unk={m:{g:float((df.loc[(df.model==m)&(df.group==g),'unknown_pieces'].sum())/df.loc[(df.model==m)&(df.group==g),'pieces'].sum())for g in groups}for m in M_NAMES}
    return {'decisions':decisions,'groups':metrics,'H31c':h,'unknown_fraction':unk,'bootstrap_replicates':nboot,'bootstrap_seed':31024},rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-zip',default=DATA_DEFAULT);ap.add_argument('--models-zip',default=MODELS_DEFAULT);ap.add_argument('--out',default=str(BASE));args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    windows,excluded,allfiles=get_windows(Path(args.data_zip))
    manifests=[{k:v for k,v in w.items() if k!='words'}for w in windows]
    writecsv(out/'cohort_manifest.csv',manifests)
    models,hashes=models_from_zip(Path(args.models_zip))
    rows,mismatches=score(windows,models)
    writecsv(out/'page_model_scores.csv',rows)
    df=pd.DataFrame(rows);issues=wide_issues(df)
    issues.to_csv(out/'issue_scores.csv',index=False)
    summary,summary_rows=compute_summary(df,issues)
    writecsv(out/'bootstrap_summary.csv',summary_rows)
    summary['audit']={'all_pages_in_target_periods':dict(allfiles),'excluded_pages':dict(excluded),'eligible_pages':len(windows),'eligible_issues':{s:len(set(w['issue'] for w in windows if w['group']==s)) for s in ['EARLY_ISIG','LATE_IMVO']},
                       'model_sha256':hashes,'additivity_mismatch_count':len(mismatches),'first_ten_mismatches':mismatches[:10],
                       'cohort_sha256':sha((out/'cohort_manifest.csv').read_bytes()),'score_sha256':sha((out/'page_model_scores.csv').read_bytes()),
                       'precommit_sha256':sha((BASE/'PRECOMMIT.md').read_bytes())}
    # Year display, diagnostic only.
    years=issues.groupby(['group','year'],as_index=False).agg(issues=('issue','nunique'),specialist_E_gain=('specialist_E_gain','mean'),specialist_retained_gain_256=('specialist_retained_gain_256','mean'))
    years.to_csv(out/'yearly_diagnostic.csv',index=False)
    (out/'RESULTS.json').write_text(json.dumps(summary,indent=2,sort_keys=True),encoding='utf8')
    print(json.dumps({'decisions':summary['decisions'],'groups':summary['groups'],'H31c':summary['H31c'],'audit':{k:v for k,v in summary['audit'].items() if k!='model_sha256'}},indent=2))

if __name__=='__main__':main()
