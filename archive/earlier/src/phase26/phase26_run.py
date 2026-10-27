#!/usr/bin/env python3
"""Three reproducible exploratory historical-isiXhosa pilots. NOT human language annotations."""
import argparse, collections, csv, hashlib, json, re
from pathlib import Path
import zipfile
import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import normalize
from textblob import TextBlob
from textblob.en import sentiment as english_sentiment_lexicon

SEED=26024
WORD=re.compile(r'[^\W\d_]+',re.UNICODE)
PAGE=re.compile(r'(IMVO|ISIG)_(\d{4})-(\d{2})-(\d{2})_p\d{3}\.txt$',re.I)
BAD={'imvo','zabantsundu','isigidimi','samaxosa'}
YEARS={1885,1887,1888}
ANCHORS=['jabavu','gqoba','makiwane','rulumente']

def clean_words(text):
    return [x for x in WORD.findall(text.lower()) if x not in BAD]

def middle(words,n):
    if len(words)<n:return []
    a=(len(words)-n)//2
    return words[a:a+n]

def dump_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with open(path,'w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def load_views(zip_path):
    pages=collections.defaultdict(dict)
    with zipfile.ZipFile(zip_path) as z:
        for inf in z.infolist():
            n=inf.filename
            if inf.is_dir() or '__MACOSX' in n or not n.endswith('.txt'):continue
            if '/XhTxt/' in n: view='xh'
            elif '/EnTxt/' in n: view='en'
            else:continue
            name=n.rsplit('/',1)[-1];m=PAGE.fullmatch(name)
            if m:
                s,y,month,day=m.groups()
                if int(y) not in YEARS:continue
                pages[name][view]=z.read(n).decode('utf-8-sig','replace').replace('\r\n','\n')
    return pages

def build(zip_path, out):
    pages=load_views(zip_path)
    windows=[]; bilingual=[];issues_by=collections.defaultdict(set)
    for file in sorted(pages):
        m=PAGE.fullmatch(file); src,y,month,day=m.groups(); year=int(y)
        p=pages[file]
        if 'xh' not in p: continue
        chunks=[b for b in re.split(r'\n\s*\n',p['xh']) if len(b.split())>=85]
        choices=[]
        for b in chunks:
            words=clean_words(b)
            if len(words)>=90:choices.append(words)
        if choices:
            tok=middle(choices[len(choices)//2],90)
            assert len(tok)==90
            # Preserve source rather than label publisher from text.
            windows.append(dict(file=file,source=src,year=year,month=int(month),issue=f'{src}_{y}-{month}-{day}',text=' '.join(tok),tokens=tok))
            issues_by[(src,year)].add(f'{src}_{y}-{month}-{day}')
        if 'en' in p:
            ew=clean_words(p['en']); xw=clean_words(p['xh'])
            if len(ew)>=80 and len(xw)>=80:
                # Page-paired views, NOT translated/parallel text.
                bilingual.append(dict(file=file,source=src,year=year,month=int(month),issue=f'{src}_{y}-{month}-{day}',en=' '.join(middle(ew,80)),xh=' '.join(middle(xw,80))))
    dupcnt=collections.Counter(x['text'] for x in windows)
    ndup=sum(dupcnt[x['text']]>1 for x in windows)
    windows=[x for x in windows if dupcnt[x['text']]==1]
    with open(out/'windows.jsonl','w',encoding='utf8') as f:
        for w in windows: f.write(json.dumps(w,ensure_ascii=False)+'\n')
    with open(out/'bilingual_pairs.jsonl','w',encoding='utf8') as f:
        for p in bilingual:f.write(json.dumps(p,ensure_ascii=False)+'\n')
    for fp in ['windows.jsonl','bilingual_pairs.jsonl']:
        (out/(fp+'.sha256')).write_text(hashlib.sha256((out/fp).read_bytes()).hexdigest()+'  '+fp+'\n')
    summary=dict(n_pages=len(pages),n_snippet_windows=len(windows),removed_exact_duplicate_windows=ndup,counts={f'{s}_{y}':len([w for w in windows if w['source']==s and w['year']==y]) for s in ('IMVO','ISIG') for y in sorted(YEARS)}, issues={f'{s}_{y}':len({w['issue'] for w in windows if w['source']==s and w['year']==y}) for s in ('IMVO','ISIG') for y in sorted(YEARS)},n_bilingual=len(bilingual),bilingual_sources=dict(collections.Counter(p['source'] for p in bilingual)))
    (out/'sample_audit.json').write_text(json.dumps(summary,indent=2))
    return windows,bilingual,summary

def chars4(tokens):
    c=collections.Counter()
    for t in tokens:
        for i in range(len(t)-3):c[t[i:i+4]]+=1
    return c

def words1(tokens):
    return collections.Counter(tokens)

def js_issue_matrix(issue_records,src,kind,rng):
    # Equal issue weighting, one source in two years; month-stratified label shuffling.
    selected=[(iss,rec) for iss,rec in issue_records.items() if rec['source']==src and rec['year'] in (1885,1888)]
    selected.sort(key=lambda t:t[0])
    if kind=='char4': docs=[chars4(rec['tokens']) for _,rec in selected]
    else: docs=[words1(rec['tokens']) for _,rec in selected]
    X=normalize(DictVectorizer(sparse=True).fit_transform(docs),norm='l1',axis=1).tocsr()
    years=np.array([rec['year'] for _,rec in selected]);months=np.array([rec['month'] for _,rec in selected])
    def score(labels):
        p=np.asarray(X[labels==1885].mean(axis=0)).ravel()
        q=np.asarray(X[labels==1888].mean(axis=0)).ravel()
        return float(jensenshannon(p,q,base=2)**2)
    obs=score(years)
    perm=[]
    for _ in range(199):
        yy=years.copy()
        for m in np.unique(months):
            ind=np.flatnonzero(months==m); yy[ind]=rng.permutation(years[ind])
        perm.append(score(yy))
    pvalue=(1+sum(x>=obs for x in perm))/(len(perm)+1)
    return dict(source=src,feature=kind,observed_js_bits=obs,permutation_95=float(np.quantile(perm,.95)),permutation_median=float(np.median(perm)),permutation_p=pvalue,n_issue_1885=int(sum(years==1885)),n_issue_1888=int(sum(years==1888))),selected

def temporal(windows,out):
    issue_records={}
    for w in windows:
        rec=issue_records.setdefault(w['issue'],dict(source=w['source'],year=w['year'],month=w['month'],tokens=[]))
        rec['tokens'].extend(w['tokens'])
    rows=[]
    for src in ['IMVO','ISIG']:
        for kind in ['char4','word1']:
            r,_=js_issue_matrix(issue_records,src,kind,np.random.default_rng(SEED + (0 if src=='IMVO' else 1)+(10 if kind=='word1' else 0)))
            rows.append(r)
    dump_csv(out/'temporal_permutation.csv',rows)
    anchor=[]
    for src in ['IMVO','ISIG']:
        for year in [1885,1888]:
            ww=[w for w in windows if w['source']==src and w['year']==year]
            count=collections.Counter(t for w in ww for t in w['tokens'])
            for a in ANCHORS:
                anchor.append(dict(source=src,year=year,anchor=a,count=count[a],per_10000_tokens=round(1e4*count[a]/max(1,sum(count.values())),3),total_tokens=sum(count.values())))
    dump_csv(out/'temporal_anchor_counts.csv',anchor)
    tops=[]
    for src in ('IMVO','ISIG'):
        counts={year:collections.Counter(t for w in windows if w['source']==src and w['year']==year for t in w['tokens']) for year in (1885,1888)}
        totals={year:sum(c.values()) for year,c in counts.items()}
        for word in set(counts[1885])|set(counts[1888]):
            if counts[1885][word]+counts[1888][word]<15:continue
            a=1e4*counts[1885][word]/totals[1885];b=1e4*counts[1888][word]/totals[1888]
            tops.append(dict(source=src,word=word,n_1885=counts[1885][word],n_1888=counts[1888][word],rate_1885=round(a,4),rate_1888=round(b,4),absolute_shift_per10k=round(abs(b-a),4)))
    tops=sorted(tops,key=lambda r:(r['source'],-r['absolute_shift_per10k']))
    dump_csv(out/'temporal_surface_strings.csv',tops[:200])
    return rows

def cluster_metric(rows,func,seed=26024,n=1000):
    # Sample independently at issue level within each source stratum; preserves within-issue page dependence.
    rng=np.random.default_rng(seed)
    group=collections.defaultdict(list)
    for r in rows:group[r['issue']].append(r)
    iss=list(group)
    stats=[]
    for _ in range(n):
        draw=rng.choice(iss,len(iss),replace=True)
        sample=[r for issue in draw for r in group[issue]]
        stats.append(func(sample))
    return [float(v) for v in np.quantile(stats,[.025,.975])]

def sentiment(bilingual,out):
    rows=[]
    for p in bilingual:
        r={k:p[k] for k in ['file','source','year','issue']}
        for lg in ['en','xh']:
            words=p[lg].split()
            r[f'{lg}_coverage']=sum(t in english_sentiment_lexicon for t in words)/len(words)
            r[f'{lg}_hits']=sum(t in english_sentiment_lexicon for t in words)
            r[f'{lg}_polarity']=float(TextBlob(p[lg]).sentiment.polarity)
            r[f'{lg}_zero']=int(abs(r[f'{lg}_polarity'])<1e-12)
        r['coverage_diff']=r['en_coverage']-r['xh_coverage']
        rows.append(r)
    dump_csv(out/'sentiment_page_diagnostics.csv',rows)
    summ=[]
    for src in ('IMVO','ISIG'):
        r=[x for x in rows if x['source']==src]
        if not r:continue
        d=dict(source=src,n_pages=len(r),n_issues=len({x['issue'] for x in r}),en_dictionary_coverage=np.mean([x['en_coverage'] for x in r]),xh_dictionary_coverage=np.mean([x['xh_coverage'] for x in r]),en_zero_polarity=np.mean([x['en_zero'] for x in r]),xh_zero_polarity=np.mean([x['xh_zero'] for x in r]),mean_english_polarity=np.mean([x['en_polarity'] for x in r]),mean_xh_polarity=np.mean([x['xh_polarity'] for x in r]),mean_coverage_diff=np.mean([x['coverage_diff'] for x in r]))
        if len({x['issue'] for x in r})>=10:
            d['coverage_difference_95ci']=cluster_metric(r,lambda x:np.mean([z['coverage_diff'] for z in x]),seed=SEED,n=1000)
        summ.append(d)
    (out/'sentiment_coverage_summary.json').write_text(json.dumps(summ,indent=2))
    return summ

def newspaper(windows,out):
    train=[w for w in windows if w['year'] in [1885,1887]]
    test=[w for w in windows if w['year']==1888]
    y=np.array([w['source']=='IMVO' for w in train],dtype=int)
    z=np.array([w['source']=='IMVO' for w in test],dtype=int)
    models=[];predrows=[]
    for name,kwargs in [('char_wb_3_5',dict(analyzer='char_wb',ngram_range=(3,5),min_df=2,max_features=30000,lowercase=True)),('word_unigram',dict(analyzer='word',ngram_range=(1,1),min_df=2,max_features=30000,lowercase=True,token_pattern=r'(?u)\b[^\W\d_]+\b'))]:
        v=TfidfVectorizer(**kwargs)
        X=v.fit_transform([w['text'] for w in train]);T=v.transform([w['text'] for w in test]);model=LogisticRegression(C=1,class_weight='balanced',random_state=SEED,max_iter=500,solver='liblinear')
        model.fit(X,y)
        p=model.predict(T);prob=model.predict_proba(T)[:,1]
        scores=[]
        for w,pred,gold,pr in zip(test,p,z,prob):
            scores.append(dict(file=w['file'],issue=w['issue'],source=w['source'],gold_imvo=int(gold),pred_imvo=int(pred),p_imvo=float(pr),correct=int(pred==gold),model=name))
        predrows.extend(scores)
        ba=float(balanced_accuracy_score(z,p))
        # Issue-level macro BA: mean issue accuracy per source.
        issue_acc=collections.defaultdict(list)
        for r in scores:issue_acc[(r['source'],r['issue'])].append(r['correct'])
        score_issue=.5*sum(np.mean(v) for (s,iss),v in issue_acc.items() if s=='IMVO')/len([1 for (s,iss) in issue_acc if s=='IMVO']) + .5*sum(np.mean(v) for (s,iss),v in issue_acc.items() if s=='ISIG')/len([1 for (s,iss) in issue_acc if s=='ISIG'])
        # Stratified issue bootstrap: robust to unequal number pages per issue and newspaper.
        rng=np.random.default_rng(SEED)
        source_issues={src:[np.mean(v) for (s,iss),v in issue_acc.items() if s==src] for src in ('IMVO','ISIG')}
        bootstrap=[.5*sum(np.mean(rng.choice(arr,len(arr),replace=True)) for arr in source_issues.values()) for _ in range(1000)]
        models.append(dict(model=name,train_pages=len(train),test_pages=len(test),train_issues=len({w['issue'] for w in train}),test_issues=len({w['issue'] for w in test}),balanced_accuracy_page=float(ba),balanced_accuracy_issue=float(score_issue),issue_bootstrap_95ci=[float(x) for x in np.quantile(bootstrap,[.025,.975])],roc_auc_page=float(roc_auc_score(z,prob)),test_source_accuracy={s:float(np.mean([r['correct'] for r in scores if r['source']==s])) for s in ('IMVO','ISIG')}))
        coef=model.coef_.ravel();features=v.get_feature_names_out();indexes=np.argsort(coef)
        dump_csv(out/f'{name}_discriminative_strings.csv',[dict(source_direction='ISIG' if i<10 else 'IMVO',feature=str(features[ind]),coefficient=float(coef[ind])) for i,ind in enumerate(list(indexes[:10])+list(indexes[-10:]))])
    dump_csv(out/'newspaper_test_predictions.csv',predrows)
    (out/'newspaper_classifier_summary.json').write_text(json.dumps(models,indent=2))
    return models

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--views-zip',default='/mnt/data/TxtLangFiles1870-1894_TB.zip');ap.add_argument('--output',default=str(Path(__file__).parent));args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    W,B,audit=build(args.views_zip,out);print('AUDIT',json.dumps(audit),flush=True)
    A=temporal(W,out);print('TEMPORAL',json.dumps(A),flush=True)
    B=sentiment(B,out);print('SENTIMENT',json.dumps(B),flush=True)
    C=newspaper(W,out);print('NEWSPAPER',json.dumps(C),flush=True)
    (out/'phase26_abc_results.json').write_text(json.dumps(dict(sample_audit=audit,temporal=A,sentiment=B,newspaper=C),indent=2))

if __name__=='__main__':main()
