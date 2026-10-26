#!/usr/bin/env python3
"""Frozen tokenizer-only data concentration test. Executes all training before evaluation."""
import collections,hashlib,json,random,sys,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path[:0]=['/mnt/data/isiXhosa_phase11_risk_certificate/reference_code','/mnt/data/isiXhosa_sampling_mechanism_phase9']
from common import iter_pages,forms,alpha_len,paragraphs,ORIGINAL_ZIP
from train_reference import sample_exact
from classic_bpe_intervene import ClassicBPE
assert hashlib.sha256((HERE/'PRECOMMIT.md').read_bytes()).hexdigest()==(HERE/'PRECOMMIT.sha256').read_text().split()[0]
ARMS=['IMVO1887','ISIG1887','IMVO1889','MODERN']
SEEDS=list(range(8501,8513))
BUDGET=200000
CORE=198000
CONDITIONS=['distributed','random_repeat_1pct','target_repeat_0p1pct','target_repeat_1pct','neutral_repeat_1pct']
M=HERE/'models';M.mkdir(exist_ok=True)
MAN=HERE/'manifest.json';CHECK=HERE/'training_choices.json';RES=HERE/'results.json'

def sha(x):return hashlib.sha256(x).hexdigest()

def build_data():
    corp={}; evals={}; docs={}; info={}
    for source,yr in [('IMVO','1887'),('ISIG','1887'),('IMVO','1889')]:
        train=[p for p in iter_pages(years=(yr,),months={f'{i:02d}' for i in range(1,10)}) if p['source']==source]
        dev=[p for p in iter_pages(years=(yr,),months={'10','11','12'}) if p['source']==source]
        key=source+yr
        corp[key]=[w for p in train for w in forms(p['text'])]
        evals[key]=[w for p in dev for w in forms(p['text'])]
        docs[key]=[(p['id']+'#'+str(i),forms(para)) for p in train for i,para in enumerate(paragraphs(p['text']))]
        info[key]={'training_pages':len(train),'eval_pages':len(dev),'train_issues':len(set(x['id'].split('_p')[0] for x in train))}
    mm=[json.loads(x) for x in Path('/mnt/data/xho.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    def is_dev(r):return int(hashlib.sha256(('phase11|'+str(r['edition'])).encode()).hexdigest(),16)%4==0
    tr=[r for r in mm if r.get('text','').strip() and not is_dev(r)]
    de=[r for r in mm if r.get('text','').strip() and is_dev(r)]
    corp['MODERN']=[w for r in tr for w in forms(r['text'])]
    evals['MODERN']=[w for r in de for w in forms(r['text'])]
    docs['MODERN']=[(str(r['edition'])+'#'+str(i),forms(para)) for r in tr for i,para in enumerate(paragraphs(r['text']))]
    info['MODERN']={'train_articles':len(tr),'eval_articles':len(de),'train_editions':len(set(r['edition'] for r in tr)),'eval_editions':len(set(r['edition'] for r in de))}
    for arm in ARMS:
        docs[arm]=[(ident,ws) for ident,ws in docs[arm] if 240<=sum(map(alpha_len,ws))<=800 and len(ws)>=20]
        if not docs[arm]:raise ValueError('No eligible training document '+arm)
        assert sum(map(alpha_len,corp[arm]))>=BUDGET
        assert len(evals[arm])>1000
        info[arm].update({'train_chars':sum(map(alpha_len,corp[arm])),'dev_chars':sum(map(alpha_len,evals[arm])),
            'train_types':len(set(corp[arm])),'dev_types':len(set(evals[arm])),'eligible_paragraphs':len(docs[arm]),
            'training_sha256':sha('\n'.join(corp[arm]).encode('utf-8'))})
    return corp,evals,docs,info

def pair_count(words):
    c=collections.Counter()
    for w in words:
        syms=['▁',*w]
        c.update(zip(syms,syms[1:]))
    return c

def repeat_fill(words,n,seed):
    arr=words*((n//sum(map(alpha_len,words))+4)*2)
    for offset in range(32):
        try:return sample_exact(arr,n,seed+offset)
        except ValueError:pass
    # Exact, finite knapsack fallback; still only uses the same repeated training paragraph.
    shuffled=list(arr);random.Random(seed).shuffle(shuffled)
    reachable=np.zeros(n+1,dtype=bool);reachable[0]=True
    parent_i=np.full(n+1,-1,dtype=int);parent_old=np.full(n+1,-1,dtype=int)
    for j,w in enumerate(shuffled):
        k=alpha_len(w)
        if k>n:continue
        prev=np.flatnonzero(reachable[:n-k+1] & ~reachable[k:])
        if len(prev):
            reachable[prev+k]=True
            parent_i[prev+k]=j
            parent_old[prev+k]=prev
        if reachable[n]:break
    if not reachable[n]:raise ValueError('Cannot construct exact repeat mass from natural paragraph')
    out=[];v=n
    while v:
        j=parent_i[v];assert j>=0
        out.append(shuffled[j]);v=parent_old[v]
    return out

def candidate_select(documents,top,second,seed):
    # Every choice is based on training data only.
    scored=[]
    for ident,ws in documents:
        c=pair_count(ws); nn=sum(map(alpha_len,ws))
        scored.append(((c[second]-c[top])/nn,ident,ws))
    targeted=max(scored,key=lambda z:(z[0],z[1]))
    neutral=min(scored,key=lambda z:(abs(z[0]),z[1]))
    rng=random.Random(seed*107+4)
    random_doc=scored[rng.randrange(len(scored))]
    return {'target':targeted,'neutral':neutral,'random':random_doc}

def training_conditions(pool,documents,seed):
    core=sample_exact(pool,CORE,seed*107+1)
    distribute=sample_exact(pool,2000,seed*107+2)
    base_pairs=pair_count(core)
    top,second=[p for p,_ in base_pairs.most_common(2)]
    chosen=candidate_select(documents,top,second,seed)
    randomws=repeat_fill(chosen['random'][2],2000,seed*107+9)
    target200=repeat_fill(chosen['target'][2],200,seed*107+10)
    target2000=repeat_fill(chosen['target'][2],2000,seed*107+10)
    neutralws=repeat_fill(chosen['neutral'][2],2000,seed*107+11)
    distributed1800=sample_exact(pool,1800,seed*107+2)
    arms={'distributed':core+distribute,
        'random_repeat_1pct':core+randomws,
        'target_repeat_0p1pct':core+distributed1800+target200,
        'target_repeat_1pct':core+target2000,
        'neutral_repeat_1pct':core+neutralws}
    for cond,ws in arms.items():
        assert sum(map(alpha_len,ws))==BUDGET, (cond, sum(map(alpha_len,ws)))
    choices={'pair_counts_top2':[[list(top),base_pairs[top]],[list(second),base_pairs[second]]],
       'candidate_scores':{key:{'training_paragraph_id':z[1],'score_second_minus_first_per_char':z[0],
           'paragraph_chars':sum(map(alpha_len,z[2]))} for key,z in chosen.items()},
       'mass_per_condition':{key:sum(map(alpha_len,ws)) for key,ws in arms.items()},
       'core_sha':sha('\n'.join(core).encode()), 'distributed_sha':sha('\n'.join(distribute).encode())}
    return arms,choices

def prepare_train():
    corp,evals,docs,info=build_data()
    manifest={'dataset_sha256':{'newspaper':sha(Path(ORIGINAL_ZIP).read_bytes()),'modern':sha(Path('/mnt/data/xho.jsonl').read_bytes())},
        'protocol_sha256':sha((HERE/'PRECOMMIT.md').read_bytes()),'budget':BUDGET,'core_chars':CORE,
        'vocab_size':2000,'seeds':SEEDS,'conditions':CONDITIONS,'info':info}
    if MAN.exists():assert json.loads(MAN.read_text())==manifest,'Manifest changed'
    else:MAN.write_text(json.dumps(manifest,indent=2))
    choices={}
    for arm in ARMS:
        for seed in SEEDS:
            t=time.monotonic()
            train_arms,choice=training_conditions(corp[arm],docs[arm],seed)
            choices[f'{arm}_{seed}']=choice
            for cond,ws in train_arms.items():
                file=M/f'{arm}_{seed}_{cond}.json'
                if file.exists():continue
                model=ClassicBPE.fit(ws,vocab_size=2000)
                dat=model.to_json();dat['train_alpha_chars']=sum(map(alpha_len,ws));dat['train_token_count']=model.train_token_count
                file.write_text(json.dumps(dat,ensure_ascii=False))
            print('TRAIN',arm,seed,'elapsed_s',round(time.monotonic()-t,2),flush=True)
    if CHECK.exists():assert json.loads(CHECK.read_text())==choices,'Choices changed'
    else:CHECK.write_text(json.dumps(choices,indent=2,ensure_ascii=False))
    print('TRAINING COMPLETE',len(list(M.glob('*.json'))))

def boot_diff(a,b,seed=20261023,n=10000):
    arr=np.array(a)-np.array(b);rng=np.random.default_rng(seed)
    samples=rng.integers(0,len(arr),size=(n,len(arr)))
    sims=np.mean(arr[samples],axis=1)
    return {'mean':float(arr.mean()),'ci95':[float(x) for x in np.quantile(sims,[.025,.975])],
        'positive_draws':int((arr>0).sum()),'negative_draws':int((arr<0).sum()),'zero_draws':int((arr==0).sum())}

def score():
    assert CHECK.exists() and MAN.exists(),'Train and freeze all choices before eval'
    assert len(list(M.glob('*.json')))==len(ARMS)*len(SEEDS)*len(CONDITIONS),'Missing trained models'
    corpus,evals,docs,info=build_data()
    final={'manifest_sha256':sha(MAN.read_bytes()),'choices_sha256':sha(CHECK.read_bytes()),
        'results':{},'stats':{},'predefined_gates':{}}
    for arm in ARMS:
        known=set(corpus[arm]);ev=collections.Counter(evals[arm]);
        md={}
        for seed in SEEDS:
            md[seed]={c:ClassicBPE.from_json(json.loads((M/f'{arm}_{seed}_{c}.json').read_text())) for c in CONDITIONS}
        shared=set.intersection(*(set(m.alphabet) for dic in md.values() for m in dic.values()))
        allowed={w:f for w,f in ev.items() if set(w)<=shared}
        unseen={w:f for w,f in allowed.items() if w not in known}
        def comp(model,ww):
            chars=sum(alpha_len(w)*f for w,f in ww.items())
            pcs=sum(model.encode_len(w)*f for w,f in ww.items())
            return {'chars':chars,'pieces':pcs,'E':100*pcs/chars if chars else None}
        armdata={}; rev=[]
        for seed in SEEDS:
            rec={}
            for cond in CONDITIONS:
                model=md[seed][cond]
                rec[cond]={'unseen':comp(model,unseen),'all':comp(model,allowed),
                    'first_merge':list(model.merges[0]),
                    'vocab_hash':sha('\n'.join(''.join(p) for p in model.merges).encode())}
            armdata[str(seed)]=rec
            rev.append(rec)
        cross={}
        for cond in CONDITIONS:
            deltas=[r[cond]['unseen']['E']-r['distributed']['unseen']['E'] for r in rev]
            cross[cond]={'paired_delta':boot_diff([r[cond]['unseen']['E'] for r in rev],[r['distributed']['unseen']['E'] for r in rev]),
                'material_harms':sum(d>=.25 for d in deltas)}
        x_tar=[r['target_repeat_1pct']['unseen']['E'] for r in rev]
        x_rand=[r['random_repeat_1pct']['unseen']['E'] for r in rev]
        cross['target_vs_random']={'paired_delta':boot_diff(x_tar,x_rand)}
        inv={'eligible_eval_occurrences':sum(allowed.values()),'eval_alpha_coverage':sum(alpha_len(w)*f for w,f in allowed.items())/sum(alpha_len(w)*f for w,f in ev.items()),
             'unseen_occurrences':sum(unseen.values()),'unseen_chars':sum(alpha_len(w)*f for w,f in unseen.items())}
        final['results'][arm]=armdata;final['stats'][arm]={'coverage':inv,'contrasts':cross}
        print('SCORE',arm,'unseen',inv['unseen_occurrences'],'target1% mean_delta',round(cross['target_repeat_1pct']['paired_delta']['mean'],4),
           'CI',cross['target_repeat_1pct']['paired_delta']['ci95'],'target-vs-random',round(cross['target_vs_random']['paired_delta']['mean'],4),flush=True)
    hits=[]
    for arm in ARMS:
        c=final['stats'][arm]['contrasts']
        a=c['target_repeat_1pct']['paired_delta'];b=c['target_vs_random']['paired_delta']
        if a['mean']>=.25 and a['ci95'][0]>0 and b['mean']>0:hits.append(arm)
    final['predefined_gates']={'passing_arms':hits,'general_effect_passed':len(hits)>=3}
    RES.write_text(json.dumps(final,indent=2,ensure_ascii=False))
    print('FINAL',final['predefined_gates'],flush=True)

if __name__=='__main__':
    op=sys.argv[1] if len(sys.argv)>1 else 'train'
    if op=='train':prepare_train()
    elif op=='score':score()
    else:raise ValueError(op)
