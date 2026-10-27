#!/usr/bin/env python3
"""Phase 11: test first-pair margin as a domain-transfer risk certificate.
Pure training-only choices are checkpointed before any scoring.
"""
import collections, hashlib, json, math, random, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
P9=Path('/mnt/data/isiXhosa_sampling_mechanism_phase9')
EX=Path('/mnt/data/isiXhosa_historical_eval/isiXhosa_external_validation')
sys.path.insert(0,str(P9 if P9.exists() else ROOT/'reference_code'))
sys.path.insert(0,str(EX if EX.exists() else ROOT/'reference_code'))
from classic_bpe_intervene import ClassicBPE
from common import training_pages, dev_pages, forms, alpha_len, ORIGINAL_ZIP
from train_reference import sample_exact

assert hashlib.sha256((ROOT/'PHASE11_PRECOMMIT.md').read_bytes()).hexdigest()==(ROOT/'PRECOMMIT.sha256').read_text().split()[0]
BUDGET=200000
SEEDS=list(range(7101,7113))
ARMS=['IMVO','ISIG','MODERN']
MODELS=ROOT/'models'; MODELS.mkdir(exist_ok=True)
MANIFEST=ROOT/'CORPUS_MANIFEST.json'
CP=ROOT/'TRAIN_ONLY_SELECTION.json'
RESULTS=ROOT/'PHASE11_RESULTS.json'

def hashbytes(x):return hashlib.sha256(x).hexdigest()

def prep():
    import collections
    pages=training_pages()
    dev=dev_pages()
    corpus={}; test={}; records={}
    for source in ('IMVO','ISIG'):
        train_records=[p for p in pages if p['source']==source]
        eval_records=[p for p in dev if p['source']==source]
        corpus[source]=[w for p in train_records for w in forms(p['text'])]
        test[source]=[w for p in eval_records for w in forms(p['text'])]
        records[source]={'train_page_files':[p['id'] for p in train_records],
                         'eval_page_files':[p['id'] for p in eval_records],
                         'train_pages':len(train_records), 'eval_pages':len(eval_records)}
    modern=[json.loads(line) for line in Path('/mnt/data/xho.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    def holdout(edition): return int(hashlib.sha256(('phase11|'+str(edition)).encode()).hexdigest(),16)%4==0
    trainmod=[r for r in modern if r.get('text','').strip() and not holdout(r['edition'])]
    testmod=[r for r in modern if r.get('text','').strip() and holdout(r['edition'])]
    corpus['MODERN']=[w for r in trainmod for w in forms(r['text'])]
    test['MODERN']=[w for r in testmod for w in forms(r['text'])]
    records['MODERN']={'train_articles':len(trainmod),'eval_articles':len(testmod),
            'train_editions':sorted(set(r['edition'] for r in trainmod)),
            'eval_editions':sorted(set(r['edition'] for r in testmod)),
            'article_salt':'sha256(phase11|edition) modulo4=0 held out'}
    assert not(set(records['MODERN']['train_editions'])&set(records['MODERN']['eval_editions']))
    for source in ARMS:
        ws=corpus[source]; val=test[source]
        assert sum(map(alpha_len,ws))>=BUDGET,(source,'too small',sum(map(alpha_len,ws)))
        assert sum(map(alpha_len,val))>30000,(source,'no evaluation data')
        known_forms=set(ws)
        records[source].update({'train_alpha_chars':sum(map(alpha_len,ws)),
                 'eval_alpha_chars':sum(map(alpha_len,val)),
                 'train_occurrences':len(ws),'eval_occurrences':len(val),
                 'train_types':len(set(ws)),
                 'unseen_eval_occurrences':sum(w not in known_forms for w in val),
                 'train_form_sha256':hashbytes('\n'.join(ws).encode())})
    manifest={'precommit_sha':(ROOT/'PRECOMMIT.sha256').read_text().split()[0],
         'input_sha256':{'newspaper_zip':hashbytes(Path(ORIGINAL_ZIP).read_bytes()),
                        'modern_xho':hashbytes(Path('/mnt/data/xho.jsonl').read_bytes())},
         'records':records,'train_chars':BUDGET,'seed_range':SEEDS,
         'vocab_size':2000,'grouping':'edition for modern, page/issue time for 1887'}
    if MANIFEST.exists():
        previous=json.loads(MANIFEST.read_text())
        assert previous==manifest,'Corpus source or split changed after freezing manifest'
    else:
        MANIFEST.write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    print('PREPARE',json.dumps({a:{k:records[a][k] for k in ['train_alpha_chars','eval_alpha_chars','unseen_eval_occurrences']} for a in ARMS}))
    return corpus,test,records

def get_data():
    return prep()[:2] if not MANIFEST.exists() else prep()[:2]

def first_pairs(sample):
    wordfreq=collections.Counter(sample)
    counts=collections.Counter()
    for w,n in wordfreq.items():
        for pair,c in collections.Counter(zip(['▁',*w],['▁',*w][1:])).items():
            counts[pair]+=n*c
    top=sorted(counts.items(),key=lambda z:(-z[1],z[0]))[:2]
    assert len(top)==2
    a,b=top[0][0],top[1][0]
    var=0
    for w,n in wordfreq.items():
        ps=collections.Counter(zip(['▁',*w],['▁',*w][1:]))
        var+=n*(ps[a]-ps[b])**2
    z=(top[0][1]-top[1][1])/math.sqrt(var) if var else float('inf')
    return top,(top[0][1]-top[1][1])/top[0][1],z

def data_name(source,seed,branch):return f'{source.lower()}_{seed}_{branch}'

def train(start,end):
    corpus,_=get_data()
    assert MANIFEST.exists()
    plans=[(source,seed) for source in ARMS for seed in SEEDS]
    rows=json.loads(CP.read_text())['rows'] if CP.exists() else []
    assert len(rows)==start,(len(rows),start)
    for ix in range(start,end):
        source,seed=plans[ix]
        sample=sample_exact(corpus[source],BUDGET,seed*101+1)
        assert sum(map(alpha_len,sample))==BUDGET
        top,margin,z=first_pairs(sample)
        candidates=[]
        for name,pair in [('greedy',top[0][0]),('second',top[1][0])]:
            t=time.monotonic()
            mod=ClassicBPE.fit(sample,vocab_size=2000,forced_prefix=[pair])
            assert mod.merges[0]==pair
            dump=mod.to_json()
            full_name=data_name(source,seed,name)
            output=MODELS/(full_name+'.json')
            output.write_text(json.dumps(dump,ensure_ascii=False))
            # train E uses exact weighted symbol-sequence total, verified on every 13th training word
            train_e=100*mod.train_token_count/BUDGET
            cs=collections.Counter(sample)
            assert abs(train_e-100*sum(cs[w]*mod.encode_len(w) for w in cs)/BUDGET)<1e-10
            candidates.append({'branch':name,'first_pair':list(pair),'file':output.name,
                  'model_sha256':hashbytes(output.read_bytes()),'train_E':train_e,
                  'fit_seconds':round(time.monotonic()-t,3), 'n_merges':len(mod.merges)})
        choice=min(candidates,key=lambda x:(x['train_E'],x['first_pair']))
        row={'source':source,'seed':seed,'sample_alpha':BUDGET,
             'sample_sha256':hashbytes('\n'.join(sample).encode()),
             'sample_unique_types':len(set(sample)),
             'pair1':list(top[0][0]),'pair2':list(top[1][0]),
             'pair_frequencies':[top[0][1],top[1][1]],
             'relative_margin':margin,'bootstrap_z_approx':z,
             'flag_at_risk':margin<=.02,'candidates':candidates,
             'selected_branch':choice['branch'],'selected_file':choice['file']}
        rows.append(row)
        CP.write_text(json.dumps({'precommit_sha':(ROOT/'PRECOMMIT.sha256').read_text().split()[0],
              'training_only':True,'rows':rows},ensure_ascii=False,indent=2))
        print('TRAIN',ix+1,'of',len(plans),source,seed,'top',list(top[0][0]),list(top[1][0]),
            'margin',round(margin,4),'flag',margin<=.02,'chosen',choice['branch'],flush=True)
    print('TRAIN_COMPLETE',end,'of',len(plans),flush=True)

def train_fullpools():
    corpus,_=get_data()
    for source in ARMS:
        path=MODELS/(f'{source.lower()}_fullpool.json')
        if path.exists():continue
        t=time.monotonic()
        mod=ClassicBPE.fit(corpus[source],vocab_size=2000)
        path.write_text(json.dumps(mod.to_json(),ensure_ascii=False))
        print('FULLPOOL',source,'secs',round(time.monotonic()-t,2),'sha',hashbytes(path.read_bytes()),flush=True)

def score():
    import scipy.stats as stats
    corpus,ev=get_data()
    checkpoint=json.loads(CP.read_text());rows=checkpoint['rows']
    assert len(rows)==len(ARMS)*len(SEEDS)
    assert checkpoint.get('training_only') is True
    rng=np.random.default_rng(20261023)
    metrics={}; dataset_summary={};allrows=[]
    for source in ARMS:
        sw=[r for r in rows if r['source']==source]
        assert len(sw)==12
        train_vocab=set(corpus[source]);val=collections.Counter(ev[source]);
        words=list(val)
        unseen=np.array([w not in train_vocab for w in words],dtype=bool)
        lens=np.array([alpha_len(w) for w in words])
        f=np.array([val[w] for w in words])
        models={}
        fullpoolname=f'{source.lower()}_fullpool.json'
        for row in sw:
            for cand in row['candidates']:
                obj=json.loads((MODELS/cand['file']).read_text())
                assert hashbytes((MODELS/cand['file']).read_bytes())==cand['model_sha256']
                models[cand['file']]=ClassicBPE.from_json(obj)
        models[fullpoolname]=ClassicBPE.from_json(json.loads((MODELS/fullpoolname).read_text()))
        # Strict per-source shared word set for all model strategies, including fullpool.
        valid=np.array([all(c in next(iter(models.values())).alphabet for c in w) for w in words],dtype=bool)
        for name,m in models.items():
            valid &= np.array([all(c in m.alphabet for c in w) for w in words],dtype=bool)
        support_unseen=(lens*f*unseen).sum()
        valid_unseen=(lens*f*valid*unseen).sum()
        coverage=float(valid_unseen/max(support_unseen,1))
        assert valid_unseen>0 and coverage>.97,(source,coverage)
        price={}
        for i,(name,m) in enumerate(models.items()):
            counts=np.array([m.encode_len(w) if valid[j] else 0 for j,w in enumerate(words)],dtype=np.int32)
            assert np.all(counts[valid]>=1)
            unseenmask=(valid & unseen)
            price[name]={'unseen_E':float(100*(counts*unseenmask*f).sum()/(lens*unseenmask*f).sum()),
                  'all_E':float(100*(counts*valid*f).sum()/(lens*valid*f).sum())}
            if (i+1)%8==0:print('SCORE',source,i+1,'of',len(models),flush=True)
        rreport=[]
        for row in sw:
            row=dict(row)
            greedy=next(c for c in row['candidates'] if c['branch']=='greedy')
            second=next(c for c in row['candidates'] if c['branch']=='second')
            gre=price[greedy['file']]['unseen_E'];sec=price[second['file']]['unseen_E']
            selected=price[row['selected_file']]['unseen_E']
            row['scores']={'greedy_unseen_E':gre,'second_unseen_E':sec,
                        'selected_unseen_E':selected,'oracle_unseen_E':min(gre,sec),
                        'greedy_all_E':price[greedy['file']]['all_E'],
                        'second_all_E':price[second['file']]['all_E'],
                        'greedy_regret_E':gre-min(gre,sec),
                        'train_selection_improvement_E':gre-selected,
                        'positive_error':gre-min(gre,sec)>=.25,
                        'selected_is_better_on_unseen':selected<=min(gre,sec)+1e-9}
            rreport.append(row);allrows.append(row)
        x=np.array([r['scores']['train_selection_improvement_E'] for r in rreport])
        draw=rng.integers(0,len(x),size=(10000,len(x)))
        intervals=np.quantile(x[draw].mean(axis=1),[.025,.975]).tolist()
        dataset_summary[source]={'train_chars':sum(map(alpha_len,corpus[source])),
           'heldout_unseen_support_alpha':int(support_unseen),
           'heldout_unseen_coverage':coverage,
           'heldout_all_coverage':float((lens*f*valid).sum()/(lens*f).sum()),
           'first_merge_dist':dict(collections.Counter(''.join(r['pair1']) for r in rreport)),
           'flag_count':sum(r['flag_at_risk'] for r in rreport),
           'material_error_count':sum(r['scores']['positive_error'] for r in rreport),
           'means':{'greedy':float(np.mean([r['scores']['greedy_unseen_E'] for r in rreport])),
              'second':float(np.mean([r['scores']['second_unseen_E'] for r in rreport])),
              'selected':float(np.mean([r['scores']['selected_unseen_E'] for r in rreport])),
              'oracle':float(np.mean([r['scores']['oracle_unseen_E'] for r in rreport])),
              'fullpool':price[fullpoolname]['unseen_E']},
           'paired_improvement_mean':float(x.mean()),
           'paired_improvement_ci95':intervals,
           'n_train_selection_oracle_agreement':sum(r['scores']['selected_is_better_on_unseen'] for r in rreport),
           'rows':rreport}
        print('SCORE_FINISH',source,'flags',dataset_summary[source]['flag_count'],
              'errors',dataset_summary[source]['material_error_count'],
              'gain',round(float(x.mean()),4),'ci',np.round(intervals,4),flush=True)
    positives=[r['scores']['positive_error'] for r in allrows]
    predicts=[r['flag_at_risk'] for r in allrows]
    tp=sum(x and y for x,y in zip(positives,predicts));fp=sum((not x) and y for x,y in zip(positives,predicts))
    fn=sum(x and (not y) for x,y in zip(positives,predicts));tn=sum((not x) and (not y) for x,y in zip(positives,predicts))
    precision=tp/(tp+fp) if tp+fp else None
    recall=tp/(tp+fn) if tp+fn else None
    by_source={src: {'different_first_pair':len(dataset_summary[src]['first_merge_dist'])>1,
                'material_errors':dataset_summary[src]['material_error_count']>0,
                'positive_improvement':dataset_summary[src]['paired_improvement_mean']>0}
               for src in ARMS}
    gate1=sum(v['different_first_pair'] and v['material_errors'] for v in by_source.values())>=2
    gate2=(precision is not None and recall is not None and precision>=.75 and recall>=.7
        and sum(v['positive_improvement'] for v in by_source.values())>=2)
    output={'protocol_sha':checkpoint['precommit_sha'],'summary':dataset_summary,
          'predictor':{'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':precision,'recall':recall,
                        'mean_regret_flagged':float(np.mean([r['scores']['greedy_regret_E'] for r in allrows if r['flag_at_risk']])) if sum(predicts) else None,
                        'mean_regret_not_flagged':float(np.mean([r['scores']['greedy_regret_E'] for r in allrows if not r['flag_at_risk']])) if sum(not x for x in predicts) else None},
          'gates':{'per_corpus':by_source,'generality_gate':gate1,'predictor_gate':gate2,'overall_pass':gate1 and gate2},
          'previously_examined_domains':True,'neural_test_run':False}
    RESULTS.write_text(json.dumps(output,indent=2,ensure_ascii=False))
    print('FINAL',json.dumps({'predictor':output['predictor'],'gates':output['gates']},indent=2),flush=True)

if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd=='prepare': prep()
    elif cmd=='train':train(int(sys.argv[2]),int(sys.argv[3]))
    elif cmd=='fullpool':train_fullpools()
    elif cmd=='score':score()
    else:raise ValueError(cmd)
