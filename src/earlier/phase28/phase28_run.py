#!/usr/bin/env python3
"""Phase 28: frozen, non-semantic lexical topic-proxy sensitivity analysis."""
import csv, hashlib, json, re, collections, argparse
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer

SEED=28024
MASK_SUBSTR=('imvo','isigidi','samaxosa','zabantsundu','bantsundu','ntsundu','mnyama','xosa','jabavu','gqoba','makiwane')
MASK_EXACT={'lovedale','mr'}
WORD=re.compile(r'[^\W\d_]+',re.U)

def write_csv(path, rows):
    if not rows: raise AssertionError('No records for '+str(path))
    with open(path,'w',encoding='utf-8',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)

def safe_words(w):
    # Exclude only publisher/person markers; do not label tokens semantically
    words=WORD.findall(w['text'].lower())
    return ' '.join(t for t in words if t not in MASK_EXACT and not any(p in t for p in MASK_SUBSTR))

def metrics_by_issue(selected, pred):
    acc=collections.defaultdict(list)
    for idx in selected:
        r=pred[idx];acc[(r['source'],r['issue'])].append(int(r['correct']))
    scores={s:np.mean([np.mean(v) for (x,_),v in acc.items() if x==s]) for s in ('IMVO','ISIG')}
    return {**{s:float(scores[s]) for s in scores},'macro':float((scores['IMVO']+scores['ISIG'])/2),
            'n_issues_imvo':sum(s=='IMVO' for s,i in acc),'n_issues_isig':sum(s=='ISIG' for s,i in acc)}

def source_benefit_by_issue(selected,cost):
    d=collections.defaultdict(list)
    for idx in selected:
        for seed, b in cost[idx].items():
            src=b['source'];delta=b['ISIG_E']-b['IMVO_E'] if src=='IMVO' else b['IMVO_E']-b['ISIG_E']
            d[(src,b['issue'],seed)].append(delta)
    out={}
    for src in ('IMVO','ISIG'):
        seedmeans=[]
        for seed in range(3):
            vals=[np.mean(v) for (s,issue,q),v in d.items() if s==src and q==seed]
            seedmeans.append(np.mean(vals))
        out[src]=float(np.mean(seedmeans))
    return out

def bootstrap_months(pairs, pred, cost, rng, nboot=2000):
    groups=collections.defaultdict(list)
    for p in pairs:groups[p['month']].append(p)
    months=sorted(groups); assert len(months)==12, months
    accvals=[];adv_i=[];adv_s=[]
    for _ in range(nboot):
        draw=rng.choice(months,len(months),replace=True)
        temp=collections.defaultdict(list);bt=collections.defaultdict(list)
        for drawi,month in enumerate(draw):
            for p in groups[month]:
                for idx in [p['i_idx'],p['s_idx']]:
                    r=pred[idx]
                    temp[(r['source'],drawi,r['issue'])].append(int(r['correct']))
                    for seed,b in cost[idx].items():
                        delta=b['ISIG_E']-b['IMVO_E'] if b['source']=='IMVO' else b['IMVO_E']-b['ISIG_E']
                        bt[(b['source'],seed,drawi,b['issue'])].append(delta)
        score=[]
        for src in ('IMVO','ISIG'):
            x=[np.mean(v) for (s,_,_),v in temp.items() if s==src]
            score.append(np.mean(x))
        accvals.append(np.mean(score))
        for src, a in [('IMVO',adv_i),('ISIG',adv_s)]:
            byseed=[np.mean([np.mean(v) for (s,q,_,_),v in bt.items() if s==src and q==seed]) for seed in range(3)]
            a.append(np.mean(byseed))
    return {'accuracy95':list(map(float,np.quantile(accvals,[.025,.975]))),
            'imvo_savings95':list(map(float,np.quantile(adv_i,[.025,.975]))),
            'isig_savings95':list(map(float,np.quantile(adv_s,[.025,.975])))}

def run(out, existing):
    src=Path(existing)
    w=[json.loads(x) for x in (src/'windows.jsonl').read_text(encoding='utf8').splitlines() if x]
    assert len(w)==581 and len({q['file'] for q in w})==581
    imap={row['file']:k for k,row in enumerate(w)}
    train_docs=[safe_words(q) for q in w if q['year'] in (1885,1887)]
    vect=TfidfVectorizer(analyzer='word',ngram_range=(1,1),min_df=2,max_df=0.35,sublinear_tf=True,
                          norm='l2',max_features=15000,token_pattern=r'(?u)\b[^\W\d_]+\b')
    vect.fit(train_docs)
    X=vect.transform([safe_words(q) for q in w]);vocab=vect.get_feature_names_out()
    # Inputs and claims retain stage provenance
    metadata=dict(n_all=len(w),n_train=sum(q['year'] in (1885,1887) for q in w),n_eval=sum(q['year']==1888 for q in w),
                  vocab_size=len(vocab),fitted_on_years=[1885,1887],match_years=[1885,1887,1888],
                  masked_word_examples=['imvo','isigidimi','jabavu'],archive_genre_labels_available=False)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    preds={name:{} for name in ['char_wb_3_5','word_unigram']}
    with open(src/'newspaper_test_predictions.csv',encoding='utf8') as f:
        for row in csv.DictReader(f):
            if row['file'] in imap:preds[row['model']][imap[row['file']]]=row
    assert len(preds['char_wb_3_5'])==207 and len(preds['word_unigram'])==207
    costs=collections.defaultdict(dict)
    with open(src/'tokenizer_page_scores.csv',encoding='utf8') as f:
        for r in csv.DictReader(f):
            if r['file'] in imap:
                r['IMVO_E']=float(r['IMVO_E']);r['ISIG_E']=float(r['ISIG_E']);costs[imap[r['file']]][int(r['seed'])]=r
    assert len(costs)==207 and all(len(v)==3 for v in costs.values())
    # Hungarian matching; exact, rectangular 1-to-1 min-cardinality pairs
    rng=np.random.default_rng(SEED);pair_rows=[];group_rows=[];rand_by_month={};quality_by_month=[];random_eval=[]
    for year in (1885,1887,1888):
        for month in range(1,13):
            ids={s:[i for i,r in enumerate(w) if r['year']==year and r['month']==month and r['source']==s] for s in ('IMVO','ISIG')}
            a,b=ids['IMVO'],ids['ISIG'];assert a and b,(year,month)
            sim=(X[a] @ X[b].T).toarray()
            row,col=linear_sum_assignment(-sim)
            assert len(row)==min(len(a),len(b))
            pairs=[]
            for j,k in zip(row,col):
                idx1,idx2=a[j],b[k];pairs.append(dict(year=year,month=month,i_idx=idx1,s_idx=idx2,
                    imvo_file=w[idx1]['file'],isig_file=w[idx2]['file'],imvo_issue=w[idx1]['issue'],isig_issue=w[idx2]['issue'],cosine=float(sim[j,k])))
            pair_rows.extend(pairs)
            # Month-and-year-specific random 1:1 *selection* and pair ordering; no dependence on model scores.
            rvals=[]
            for z in range(500):
                i_sel=rng.permutation(a)[:len(pairs)]
                s_sel=rng.permutation(b)[:len(pairs)]
                rvals.append((float(np.mean([sim[a.index(ii),b.index(ss)] for ii,ss in zip(i_sel,s_sel)])),list(i_sel),list(s_sel)))
            randmean=float(np.mean([r[0] for r in rvals]))
            rand_by_month[(year,month)]=rvals
            group_rows.append(dict(year=year,month=month,imvo_eligible=len(a),isig_eligible=len(b),n_pairs=len(pairs),
               matched_mean_cos=float(np.mean([p['cosine'] for p in pairs])),random_mean_cos=randmean,
               similarity_gain=float(np.mean([p['cosine'] for p in pairs]))-randmean,
               matched_median_cos=float(np.median([p['cosine'] for p in pairs])),fraction_matched_sim_ge_0_1=float(np.mean([p['cosine']>=.1 for p in pairs]))))
    write_csv(out/'all_year_matched_pairs.csv',[{k:v for k,v in p.items() if k not in ('i_idx','s_idx')} for p in pair_rows])
    write_csv(out/'monthly_matching_quality.csv',group_rows)
    pairs88=[p for p in pair_rows if p['year']==1888]
    sel88=[idx for p in pairs88 for idx in (p['i_idx'],p['s_idx'])]
    assert len(set(sel88))==len(sel88) and len(sel88)==2*sum(q['year']==1888 for q in group_rows for _ in range(q['n_pairs']))
    outcomes={}
    for model,pred in preds.items():
        matched=metrics_by_issue(sel88,pred)
        full=metrics_by_issue(list(pred),pred)
        # Full random one-to-one match sample for 500 global repetitions
        random_acc=[]
        for rep in range(500):
            ids=[]
            for month in range(1,13):
                _,isamp,ssamp=rand_by_month[(1888,month)][rep]
                ids.extend(list(isamp)+list(ssamp))
            random_acc.append(metrics_by_issue(ids,pred)['macro'])
        outcomes[model]={'matched':matched,'full':full,'random_mean':float(np.mean(random_acc)),
                        'random_sd':float(np.std(random_acc,ddof=1)),
                        'random_95percentile':list(map(float,np.quantile(random_acc,[.025,.975]))),
                        'matched_minus_random_mean':float(matched['macro']-np.mean(random_acc))}
    bpe_matched=source_benefit_by_issue(sel88,costs)
    bpe_full=source_benefit_by_issue(list(costs),costs)
    bpe_random=[]
    for rep in range(500):
        ids=[]
        for month in range(1,13):
            _,isamp,ssamp=rand_by_month[(1888,month)][rep]
            ids.extend(list(isamp)+list(ssamp))
        bpe_random.append(source_benefit_by_issue(ids,costs))
    rng2=np.random.default_rng(SEED+1)
    ci=bootstrap_months(pairs88,preds['char_wb_3_5'],costs,rng2)
    m88=[g for g in group_rows if g['year']==1888]
    # Weighted by matched-pair count, average observed gain per matched pair with 12-month cluster bootstrap.
    observed_delta=sum(g['n_pairs']*g['similarity_gain'] for g in m88)/sum(g['n_pairs'] for g in m88)
    gains=[]
    for _ in range(2000):
        sample=rng2.choice(len(m88),len(m88),replace=True)
        gains.append(sum(m88[k]['n_pairs']*m88[k]['similarity_gain'] for k in sample)/sum(m88[k]['n_pairs'] for k in sample))
    gain95=list(map(float,np.quantile(gains,[.025,.975])))
    results={
      'scientific_scope':'Lexical-content proxy sensitivity, NOT validated genre/article-type control',
      'n_matched_pairs_1888':len(pairs88),'n_matched_issue_imvo':outcomes['char_wb_3_5']['matched']['n_issues_imvo'],
      'n_matched_issue_isig':outcomes['char_wb_3_5']['matched']['n_issues_isig'],
      'match_quality_1888':{'cosine_gain_over_random':float(observed_delta),'month_bootstrap95':gain95,
        'matched_cosine':float(sum(g['n_pairs']*g['matched_mean_cos'] for g in m88)/sum(g['n_pairs'] for g in m88)),
        'random_cosine':float(sum(g['n_pairs']*g['random_mean_cos'] for g in m88)/sum(g['n_pairs'] for g in m88))},
      'classification':outcomes,
      'bpe':{'matched_savings':bpe_matched,'full_savings':bpe_full,
        'random_means':{s:float(np.mean([p[s] for p in bpe_random])) for s in ('IMVO','ISIG')},
        'random95':{s:list(map(float,np.quantile([p[s] for p in bpe_random],[.025,.975]))) for s in ('IMVO','ISIG')},
        'month_bootstrap95':{'IMVO':ci['imvo_savings95'],'ISIG':ci['isig_savings95']}},
      'matched_classification_month_bootstrap95':ci['accuracy95']}
    h28a=observed_delta>=.025 and gain95[0]>0
    h28b=outcomes['char_wb_3_5']['matched']['macro']>.65 and ci['accuracy95'][0]>.50
    h28c=ci['imvo_savings95'][0]>0 and ci['isig_savings95'][0]>0
    results['preregistered_outcomes']={'H28a':h28a,'H28b':h28b,'H28c':h28c}
    (out/'RESULTS.json').write_text(json.dumps(results,indent=2))
    # Input/index recalc includes all individual paired and model score data for validation
    write_csv(out/'matched_1888_page_scores.csv',[
      {'file':w[idx]['file'],'issue':w[idx]['issue'],'month':w[idx]['month'],'source':w[idx]['source'],
       'cosine_to_pair':p['cosine'],'paired_file':w[p['s_idx'] if idx==p['i_idx'] else p['i_idx']]['file'],
       'char_source_correct':int(preds['char_wb_3_5'][idx]['correct']),
       'char_source_p_imvo':float(preds['char_wb_3_5'][idx]['p_imvo']),
       'word_source_correct':int(preds['word_unigram'][idx]['correct']),
       **{f'bpe_advantage_seed{seed}':float((costs[idx][seed]['ISIG_E']-costs[idx][seed]['IMVO_E']) if w[idx]['source']=='IMVO' else (costs[idx][seed]['IMVO_E']-costs[idx][seed]['ISIG_E'])) for seed in range(3)}}
      for p in pairs88 for idx in [p['i_idx'],p['s_idx']]])
    return results

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prior',default='/mnt/data/isiXhosa_phase26_culture_pilots');ap.add_argument('--out',default=str(Path(__file__).parent));args=ap.parse_args()
    r=run(Path(args.out),args.prior)
    print(json.dumps(r,indent=2))
