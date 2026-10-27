#!/usr/bin/env python3
"""Phase 27: Source-to-source diachronic prediction with controlled BPE vocabularies.

Prospectively specified on 2026-10-24. Exploratory only: these 1888 texts
have been analyzed in prior phases. No cultural meaning labels are inferred.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import sentencepiece as spm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

TRAIN_YEAR = 1887
TEST_YEARS = (1885,1888)
SOURCES = ('IMVO', 'ISIG')
SEEDS = (27,28,29)
CHARS_PER_TOKENIZER = 40000
N_BOOT = 2000
WORD_SUBSTR = ('imvo','isigidi','samaxosa','zabantsundu','bantsundu',
               'ntsundu','mnyama','xosa','jabavu','gqoba','makiwane')
WORD_EXACT = {'mr','sir','lovedale','b','g','e','th'}

def read_jsonl(path):
    return [json.loads(line) for line in open(path,encoding='utf-8')]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def csv_write(path, rows, cols=None):
    rows = list(rows)
    if cols is None:
        if not rows: raise RuntimeError(f'cannot write zero rows: {path}')
        cols=list(rows[0])
    with open(path,'w',encoding='utf-8',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)

def get_text(w, masked):
    if not masked:
        return w['text']
    return ' '.join(t for t in w['tokens']
                    if t not in WORD_EXACT and not any(x in t for x in WORD_SUBSTR))

def alphabetic(t):
    return sum(c.isalpha() for c in t)

def pick_chars(wins, source, seed, budget):
    ww=[w for w in wins if w['source']==source and w['year']==TRAIN_YEAR]
    assert len(ww)>0
    rng=np.random.default_rng(seed + (500 if source=='IMVO' else 100))
    ww=[ww[i] for i in rng.permutation(len(ww))]
    lines=[]
    use=[]
    chars=0
    n_cut=0
    for w in ww:
        words=[]
        for token in w['tokens']:
            n=alphabetic(token)
            if chars+n <= budget:
                words.append(token); chars+=n
            else:
                rest=budget-chars
                if rest:
                    shortened=''.join(c for c in token if c.isalpha())[:rest]
                    words.append(shortened);chars+=rest;n_cut+=1
                break
        if words:
            lines.append(' '.join(words))
            use.append(w['issue'])
        if chars >= budget: break
    assert chars==budget,(source,seed,chars,budget)
    return lines, use, n_cut

def fit_tokenizers(windows,output):
    audit=[]
    for seed in SEEDS:
        selected={}
        for source in SOURCES:
            for chars in (CHARS_PER_TOKENIZER//2,CHARS_PER_TOKENIZER):
                selected[source,chars]=pick_chars(windows,source,seed,chars)
        for imvo_share in (0,50,100):
            pair=[]; included=[]; trunc=0
            if imvo_share==0:
                s,n,c=selected['ISIG',CHARS_PER_TOKENIZER]
                pair+=s;included+=n;trunc+=c
            elif imvo_share==100:
                s,n,c=selected['IMVO',CHARS_PER_TOKENIZER]
                pair+=s;included+=n;trunc+=c
            else:
                for source in SOURCES:
                    s,n,c=selected[source,CHARS_PER_TOKENIZER//2]
                    pair+=s;included+=n;trunc+=c
            input=output/f'fit_1887_imvo{imvo_share}_seed{seed}.txt'
            input.write_text('\n'.join(pair)+'\n',encoding='utf-8')
            assert sum(alphabetic(l) for l in pair)==CHARS_PER_TOKENIZER
            prefix=output/f'BPE_imvo{imvo_share}_seed{seed}'
            spm.SentencePieceTrainer.train(
                input=str(input), model_prefix=str(prefix),
                model_type='bpe', vocab_size=1000, character_coverage=1.0,
                normalization_rule_name='identity', hard_vocab_limit=False,
                unk_id=0,bos_id=-1,eos_id=-1,pad_id=-1,
                shuffle_input_sentence=False)
            sp=spm.SentencePieceProcessor(model_file=str(prefix)+'.model')
            audit.append(dict(seed=seed,imvo_share=imvo_share,year_fit=TRAIN_YEAR,
                              train_chars=CHARS_PER_TOKENIZER,n_lines=len(pair),
                              n_unique_issues=len(set(included)),
                              n_truncated_tokens=trunc,
                              actual_vocab_size=sp.vocab_size(),
                              input_sha256=sha(input),model_sha256=sha(str(prefix)+'.model')))
    csv_write(output/'tokenizer_fit_audit.csv',audit)
    print('TOKENIZER_MODELS',len(audit),flush=True)
    return audit

def make_features(method, train, test, sp):
    if method=='char':
        vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),
                            min_df=2,max_features=30000,lowercase=True)
        return vec.fit_transform(train),vec.transform(test),vec.get_feature_names_out().size
    if method=='word':
        vec=TfidfVectorizer(analyzer='word',token_pattern=r'(?u)\b\w+\b',
                            min_df=2,max_features=30000,lowercase=True)
        return vec.fit_transform(train),vec.transform(test),vec.get_feature_names_out().size
    assert method=='bpe'
    transform=lambda texts:[' '.join(sp.encode(s,out_type=str)) for s in texts]
    vec=TfidfVectorizer(analyzer='word',lowercase=False,token_pattern=r'(?u)\S+',
                        min_df=2,max_features=30000)
    return vec.fit_transform(transform(train)),vec.transform(transform(test)),vec.get_feature_names_out().size

def eval_classifiers(windows,output):
    data=[w for w in windows if w['year'] in TEST_YEARS]
    assert len({w['issue'] for w in data}) == len(set(w['issue'] for w in data))
    assert set(w['year'] for w in data)==set(TEST_YEARS)
    audit=[]; results=[]
    models=[('char',-1,None,None),('word',-1,None,None)]
    for seed in SEEDS:
        for mixture in (0,50,100):
            fp=output/f'BPE_imvo{mixture}_seed{seed}.model'
            if not fp.exists():raise FileNotFoundError(fp)
            models.append(('bpe',seed,mixture,spm.SentencePieceProcessor(model_file=str(fp))))
    for masked in (False,True):
        for fold in range(4):
            train_months=[m for m in range(1,13) if (m-1)%4 != fold]
            test_months=[m for m in range(1,13) if (m-1)%4 == fold]
            alltest=[w for w in data if w['month'] in test_months]
            for trsource in SOURCES:
                train=[w for w in data if w['source']==trsource and w['month'] in train_months]
                assert not ({w['issue'] for w in train}&{w['issue'] for w in alltest})
                traintexts=[get_text(w,masked) for w in train]
                testtexts=[get_text(w,masked) for w in alltest]
                y=np.array([int(w['year']==1888) for w in train])
                assert len(set(y))==2
                for kind,seed,mixture,sp in models:
                    name=kind if kind!='bpe' else f'bpe{mixture}'
                    X,Z,n_feat=make_features(kind,traintexts,testtexts,sp)
                    model=LogisticRegression(C=1,solver='liblinear',class_weight='balanced',
                                             random_state=27,max_iter=500).fit(X,y)
                    pred=model.predict(Z).astype(int)
                    prob=model.predict_proba(Z)[:,list(model.classes_).index(1)]
                    audit.append(dict(masked=int(masked),fold=fold,
                                      train_source=trsource,model=name,seed=seed,
                                      n_features=n_feat,n_train=len(train),
                                      n_train_1885=int(sum(y==0)),n_train_1888=int(sum(y==1)),
                                      n_test=len(alltest),n_test_1885=sum(w['year']==1885 for w in alltest),
                                      n_test_1888=sum(w['year']==1888 for w in alltest)))
                    for w,pr,p in zip(alltest,pred,prob):
                        results.append(dict(file=w['file'],issue=w['issue'],
                                            test_source=w['source'],year=w['year'],
                                            month=w['month'],fold=fold,train_source=trsource,
                                            masked=int(masked),model=name,seed=seed,
                                            prediction_year=1888 if pr else 1885,
                                            prob_1888=float(p),correct=int(pr==int(w['year']==1888))))
            print('PREDICTION_FOLD',int(masked),fold,'n_predictions',len(results),flush=True)
    csv_write(output/'prediction_audit.csv',audit)
    csv_write(output/'period_predictions.csv',results)
    return audit,results

def issue_mean(records):
    r=collections.defaultdict(list)
    for p in records:
        r[(p['test_source'],p['year'],p['issue'])].append(int(p['correct']))
    return [dict(test_source=s,year=y,issue=iss,
                 month=int(iss.split('_')[1][5:7]),accuracy=float(np.mean(v)))
            for (s,y,iss),v in sorted(r.items())]

def accuracy_from_issues(rows, test_source):
    # all issues equal within each of two years
    d=[]
    for year in TEST_YEARS:
        sc=[p['accuracy'] for p in rows if p['test_source']==test_source and p['year']==year]
        assert len(sc)>0
        d.append(float(np.mean(sc)))
    return .5*sum(d)

def accuracy_month_sample(rows, src, sampled_months):
    values=[]
    for y in TEST_YEARS:
        # Each sampled month contributes all source/year issues (possibly >1).
        vals=[]
        for m in sampled_months:
            vals.extend(p['accuracy'] for p in rows
                        if p['test_source']==src and p['year']==y and p['month']==m)
        assert vals
        values.append(float(np.mean(vals)))
    return .5*sum(values)

def summarize_eval(predictions,output):
    # Reduce BPE repeats before bootstrap. Three tokenizer seeds are treated as
    # fixed-condition replications, not 3 independent newspaper samples.
    grp=collections.defaultdict(list)
    for p in predictions:
        grp[(p['masked'],p['model'],p['seed'],p['train_source'])].append(p)
    allscores=[]
    issgrp={}
    for (masked,method,seed,trsrc),rp in sorted(grp.items()):
        agg=issue_mean(rp)
        issgrp[masked,method,seed,trsrc]=agg
        for source in SOURCES:
            allscores.append(dict(masked=masked,model=method,seed=seed,
                                  train_source=trsrc,test_source=source,
                                  issue_balanced_accuracy=accuracy_from_issues(agg,source),
                                  page_balanced_accuracy=.5*sum(np.mean([x['correct'] for x in rp
                                                            if x['test_source']==source and x['year']==y])
                                                            for y in TEST_YEARS),
                                  n_test_issues=len(set(x['issue'] for x in rp if x['test_source']==source)),
                                  n_test_pages=len([x for x in rp if x['test_source']==source])))
    csv_write(output/'cell_scores_by_seed.csv',allscores)
    avg=collections.defaultdict(list)
    for a in allscores:
        avg[(a['masked'],a['model'],a['train_source'],a['test_source'])].append(a['issue_balanced_accuracy'])
    macro=[]
    for (mask,model,tr,te),vals in sorted(avg.items()):
        macro.append(dict(masked=mask,model=model,train_source=tr,
                          test_source=te,issue_balanced_accuracy=float(np.mean(vals)),
                          seed_scores=json.dumps([float(x) for x in vals])))
    csv_write(output/'cell_scores.csv',macro)

    # Cache 12-month sufficient statistics for issue-weighted paired bootstrap.
    # This does not alter any protocol; it avoids repeatedly scanning Python rows.
    month_cache={}
    for (mask,method,seed,trsrc),issue_rows in issgrp.items():
        for source in SOURCES:
            numer=np.zeros((2,12),dtype=float)
            denom=np.zeros((2,12),dtype=float)
            for r in issue_rows:
                if r['test_source']!=source: continue
                yi=0 if r['year']==1885 else 1
                mm=r['month']-1
                numer[yi,mm]+=r['accuracy']
                denom[yi,mm]+=1
            assert np.all(denom>0)
            month_cache[mask,method,seed,trsrc,source]=(numer,denom)
    def lookup(mask,model,tr,te):
        return float(np.mean([r['issue_balanced_accuracy'] for r in allscores
                              if (r['masked'],r['model'],r['train_source'],r['test_source'])
                               == (mask,model,tr,te)]))
    def get_month_score(mask,method,seed,tr,te,month_weights):
        numer,denom=month_cache[mask,method,seed,tr,te]
        return .5*sum(float(np.dot(numer[yi],month_weights)/np.dot(denom[yi],month_weights))
                      for yi in (0,1))
    def metric_cells(cells):
        within=.5*(cells['IMVO','IMVO']+cells['ISIG','ISIG'])
        cross=.5*(cells['IMVO','ISIG']+cells['ISIG','IMVO'])
        return within,cross,within-cross
    summary=[]
    rng=np.random.default_rng(27001)
    # Pair every method within the same bootstrap month sample for accurate paired contrasts.
    month_draws=[np.bincount(rng.integers(0,12,size=12),minlength=12) for _ in range(N_BOOT)]
    for mask in (0,1):
        for method in ['char','word','bpe0','bpe50','bpe100']:
            variants=sorted(set(x['seed'] for x in allscores if x['masked']==mask and x['model']==method))
            cells={(tr,te):lookup(mask,method,tr,te) for tr in SOURCES for te in SOURCES}
            wi,cr,gap=metric_cells(cells)
            bootstrap=[]
            for draw in month_draws:
                c={}
                for tr in SOURCES:
                    for te in SOURCES:
                        c[tr,te]=float(np.mean([
                            get_month_score(mask,method,seed,tr,te,draw)
                            for seed in variants]))
                bootstrap.append(metric_cells(c))
            bs=np.asarray(bootstrap)
            summary.append(dict(masked=mask,model=method,within=wi,cross=cr,
                                transfer_gap=gap,transfer_gap_ci_low=float(np.quantile(bs[:,2],.025)),
                                transfer_gap_ci_high=float(np.quantile(bs[:,2],.975)),
                                within_ci_low=float(np.quantile(bs[:,0],.025)),
                                within_ci_high=float(np.quantile(bs[:,0],.975)),
                                cross_ci_low=float(np.quantile(bs[:,1],.025)),
                                cross_ci_high=float(np.quantile(bs[:,1],.975))))
    csv_write(output/'transfer_summary.csv',summary)
    # Paired corpus-mix contrast for each cross-domain transfer direction.
    contrasts=[]
    for mask in (0,1):
        for tr,te in [('IMVO','ISIG'),('ISIG','IMVO')]:
            v=lookup(mask,'bpe100',tr,te)-lookup(mask,'bpe0',tr,te)
            bs=[]
            for months in month_draws:
                pair=[]
                for seed in SEEDS:
                    a=get_month_score(mask,'bpe100',seed,tr,te,months)
                    b=get_month_score(mask,'bpe0',seed,tr,te,months)
                    pair.append(a-b)
                bs.append(np.mean(pair))
            contrasts.append(dict(masked=mask,train_source=tr,test_source=te,
                                  bpe100_minus_bpe0=v,
                                  ci_low=float(np.quantile(bs,.025)),
                                  ci_high=float(np.quantile(bs,.975))))
    csv_write(output/'bpe_causal_contrasts.csv',contrasts)
    base=[x for x in summary if x['masked']==0 and x['model']=='char'][0]
    H27a=lookup(0,'char','IMVO','IMVO')>.60 and lookup(0,'char','ISIG','ISIG')>.60
    H27b=base['transfer_gap']>=.10 and base['transfer_gap_ci_low']>0
    H27c=any(abs(c['bpe100_minus_bpe0'])>=.05 and
             (c['ci_low']>0 or c['ci_high']<0) for c in contrasts if c['masked']==0)
    hypotheses=[dict(id='H27a',supported=bool(H27a),criterion='Both char within-source BA > 0.60'),
                dict(id='H27b',supported=bool(H27b),criterion='Char gap >= 0.10 and month CI above 0'),
                dict(id='H27c',supported=bool(H27c),criterion='Either cross-source BPE endpoint shift abs >= 0.05 and CI excludes 0')]
    csv_write(output/'hypothesis_tests.csv',hypotheses)
    return summary,contrasts,hypotheses

def descriptive(windows,output):
    audit=[];rr=[]
    for seed in SEEDS:
        sp={m:spm.SentencePieceProcessor(model_file=str(output/f'BPE_imvo{m}_seed{seed}.model'))
            for m in (0,50,100)}
        for source in SOURCES:
            for year in TEST_YEARS:
                ww=[w for w in windows if w['source']==source and w['year']==year]
                for mask in (False,True):
                    chars=sum(alphabetic(get_text(w,mask)) for w in ww)
                    r=dict(seed=seed,source=source,year=year,masked=int(mask),n_pages=len(ww),
                           chars=chars)
                    for mix,model in sp.items():
                        r[f'pieces_imvo{mix}']=sum(len(model.encode(get_text(w,mask),out_type=int)) for w in ww)
                        r[f'E_imvo{mix}']=100*r[f'pieces_imvo{mix}']/chars
                    audit.append(r)
    csv_write(output/'bpe_cost_by_source_year.csv',audit)
    for source in SOURCES:
        for year in [1885,1887,1888]:
            words=[t for w in windows if w['source']==source and w['year']==year for t in w['tokens']]
            for root,exact in [('ntsundu','abantsundu'),('mnyama','abamnyama')]:
                n=sum(root in w for w in words)
                e=sum(w==exact for w in words)
                rr.append(dict(source=source,year=year,root=root,exact=exact,
                               n_words=len(words),substring_occurrences=n,
                               exact_word_occurrences=e,
                               substring_per_10000=10000*n/len(words),
                               exact_per_10000=10000*e/len(words)))
    csv_write(output/'published_root_string_diagnostics.csv',rr)
    return audit,rr

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--windows',default='/mnt/data/isiXhosa_phase26_culture_pilots/windows.jsonl')
    ap.add_argument('--output',default=str(Path(__file__).parent))
    ap.add_argument('--stage',choices=['fit','eval','summary','all'],default='all')
    args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    windows=read_jsonl(args.windows)
    assert len(windows)==581,('unexpected source dataset length',len(windows))
    assert len({w['file'] for w in windows})==len(windows)
    assert all(w['year'] in (1885,1887,1888) for w in windows)
    if args.stage in ('fit','all'):
        fit_tokenizers(windows,out)
    if args.stage in ('eval','all'):
        audit,pred=eval_classifiers(windows,out)
        print('N_FITS',len(audit),'N_PREDICTIONS',len(pred),flush=True)
    if args.stage in ('summary','all'):
        if args.stage=='summary':
            pred=list(csv.DictReader(open(out/'period_predictions.csv',encoding='utf8')))
            for p in pred:
                for k in ['masked','year','month','fold','seed','correct']:
                    p[k]=int(p[k])
                p['prob_1888']=float(p['prob_1888'])
        sums,contrasts,hyp=summarize_eval(pred,out)
        costs,roots=descriptive(windows,out)
        (out/'results.json').write_text(json.dumps(dict(hypotheses=hyp,
                    transfer=sums,bpe_contrasts=contrasts,
                    n_windows=len(windows),n_supervised_windows=sum(w['year'] in TEST_YEARS for w in windows),
                    bpe_training_only_year=TRAIN_YEAR,source_sha256=sha(args.windows)),indent=2))
        print('HYPOTHESES',json.dumps(hyp),flush=True)
        print('CHAR_TRANSFER',json.dumps([x for x in sums if x['model']=='char' and x['masked']==0]),flush=True)
        print('BPE_CONTRAST',json.dumps([x for x in contrasts if x['masked']==0]),flush=True)
if __name__=='__main__':
    main()
