#!/usr/bin/env python3
"""Phase 8: test tokenizer-data sampling strategies on retrospective validation.
All treatments and endpoints are defined in PRECOMMIT.md.
"""
import collections, hashlib, json, math, random, statistics, sys, tempfile, time
from pathlib import Path
import numpy as np
import sentencepiece as spm

ROOT=Path(__file__).resolve().parent
EX=Path('/mnt/data/isiXhosa_historical_eval/isiXhosa_external_validation')
sys.path.insert(0,str(EX))
from common import training_pages, dev_pages, forms, alpha_len
from train_reference import BUDGET, sample_exact
assert BUDGET==421696
MODELS=ROOT/'models'; MODELS.mkdir(exist_ok=True)
METHODS=('word_uniform','issue_proportional','issue_equal')
SEEDS=tuple(range(1001,1021))
MIXES=(0.5,1.0)
OUT=ROOT/'results.json'
REF=EX/'models'
START=time.monotonic()


def issue_data():
    pages=training_pages(); data={s:collections.defaultdict(list) for s in ('IMVO','ISIG')}
    for p in pages:
        issue=f"{p['source']}_{p['year']}-{p['month']}-{p['day']}"
        data[p['source']][issue].extend(forms(p['text']))
    for d in data:
        assert all(v for v in data[d].values())
        data[d]=dict(sorted(data[d].items()))
    return data


def apportioned(available,budget,equal):
    ids=sorted(available)
    a=np.array([available[z] for z in ids],dtype=np.int64)
    target=(np.full(len(a),budget/len(a)) if equal else budget*a/a.sum())
    # budget is less than sum of issue capacities, no capacity binds at selected budgets
    base=np.floor(target).astype(int)
    remainder=budget-int(base.sum())
    fraction=target-base
    order=sorted(range(len(ids)),key=lambda i:(-fraction[i],ids[i]))
    for i in order[:remainder]:base[i]+=1
    assert sum(base)==budget
    assert all(x<=avail for x,avail in zip(base,a)), 'quota infeasible; do not reassign selectively'
    return {z:int(v) for z,v in zip(ids,base)}


def sample_indexed_occurrences(ww,needed,seed):
    assert 0<=needed<=sum(alpha_len(w) for w in ww)
    if needed==0:return [],[]
    rng=random.Random(seed); seq=list(range(len(ww))); rng.shuffle(seq)
    out=[];indexes=[];rem=needed
    for i in seq:
        n=alpha_len(ww[i])
        if 0<n<=rem:
            out.append(ww[i]);indexes.append(i);rem-=n
        if not rem:break
    if rem:raise RuntimeError(f'cannot fill exact quota {needed}, remaining {rem}')
    assert len(indexes)==len(set(indexes))
    assert sum(map(alpha_len,out))==needed
    return out,indexes


def train_one(data,method,mix,seed):
    budgets={'IMVO':round(BUDGET*mix),'ISIG':BUDGET-round(BUDGET*mix)}
    selected=[]; allocations={}; issue_occ={}
    for source in ('IMVO','ISIG'):
        budget=budgets[source]
        if not budget:continue
        source_data=data[source]; available={id:sum(map(alpha_len,w)) for id,w in source_data.items()}
        if method=='word_uniform':
            # Preserve the original sampler exactly for this arm.
            source_words=[w for p in training_pages_cached if p['source']==source for w in forms(p['text'])]
            sample=sample_exact(source_words,budget,seed*101+(1 if source=='IMVO' else 2))
            # Multisets of occurrences allocated back to issues using the sampled type count,
            # avoiding a false assertion about exact identity of repeated word instances.
            unallocated=collections.Counter(sample)
            per_issue={}
            for issue,ww in source_data.items():
                cs=collections.Counter(ww)
                part=sum(min(v,unallocated[w]) * alpha_len(w) for w,v in cs.items())
                for w,v in cs.items():unallocated[w]-=min(v,unallocated[w])
                per_issue[issue]=part
            assert not any(v>0 for v in unallocated.values())
            # Per-issue sample allocation inferred from types is *not reliable* when the
            # same word occurs in multiple issues. Thus do not analyze raw arm's per-issue
            # allocation with this approximate mapping; report as unavailable.
            allocations[source]=None
            issue_occ[source]=None
        else:
            quotas=apportioned(available,budget,equal=(method=='issue_equal'))
            sample=[];occ={}
            for issue,ww in source_data.items():
                q=quotas[issue]
                if not q:continue
                digits=int(hashlib.sha256(f'{seed}|{method}|{source}|{issue}'.encode()).hexdigest()[:12],16)
                chosen,ind=sample_indexed_occurrences(ww,q,digits)
                sample.extend(chosen);occ[issue]=len(ind)
            allocations[source]=quotas
            issue_occ[source]=occ
        selected.extend(sample)
        assert sum(map(alpha_len,sample))==budget
    assert sum(map(alpha_len,selected))==BUDGET
    random.Random(seed*101+3).shuffle(selected)
    name=f'{method}_mix{int(mix*100):03d}_draw{seed}'
    prefix=MODELS/name
    with tempfile.TemporaryDirectory() as td:
        inp=Path(td)/'input.txt'
        with inp.open('w',encoding='utf8') as out:
            for i in range(0,len(selected),70):out.write(' '.join(selected[i:i+70])+'\n')
        spm.SentencePieceTrainer.train(input=str(inp),model_prefix=str(prefix),
             model_type='bpe',vocab_size=2000,character_coverage=1.0,
             normalization_rule_name='identity',num_threads=2,hard_vocab_limit=False,
             bos_id=-1,eos_id=-1,pad_id=-1,unk_id=0,max_sentence_length=15000,
             shuffle_input_sentence=False,minloglevel=2)
    vocab_sha=hashlib.sha256((MODELS/(name+'.vocab')).read_bytes()).hexdigest()
    return {'name':name,'seed':seed,'method':method,'mix':mix,
        'char_budget':BUDGET,'source_budgets':budgets,'unique_sampled_types':len(set(selected)),
        'vocab_sha256':vocab_sha,'issue_allocation':allocations,'issue_sample_occurrences':issue_occ}


def evaluate(results,train):
    # Validation never informs treatments. All processors must score IDENTICAL form support.
    trseen={w for p in train for w in forms(p['text'])}
    vals={s:collections.Counter(w for p in dev_pages() if p['source']==s for w in forms(p['text'])) for s in ('IMVO','ISIG')}
    seen_count={s:{w:n for w,n in counter.items() if w in trseen} for s,counter in vals.items()}
    unseen_count={s:{w:n for w,n in counter.items() if w not in trseen} for s,counter in vals.items()}
    alphabet=set(c for p in train for w in forms(p['text']) for c in w if c.isalpha())
    eligible={s:{w:all((not c.isalpha() or c in alphabet) for c in w) for w in vals[s]} for s in vals}
    # Intersection with all fitted models to exclude any source-specific UNK token advantage.
    for ix,rr in enumerate(results):
        p=spm.SentencePieceProcessor(model_file=str(MODELS/(rr['name']+'.model')))
        uid=p.unk_id()
        for s in vals:
            for w in vals[s]:
                if eligible[s][w] and uid in p.encode(w,out_type=int):eligible[s][w]=False
        if (ix+1)%20==0:print('COMMON_SUPPORT',ix+1,'/',len(results),flush=True)
    actual={s:collections.Counter({w:n for w,n in vals[s].items() if eligible[s][w]}) for s in vals}
    effective={s:{'total_words':sum(vals[s].values()),'valid_words':sum(actual[s].values()),
                  'total_alpha':sum(n*alpha_len(w) for w,n in vals[s].items()),
                  'valid_alpha':sum(n*alpha_len(w) for w,n in actual[s].items()),
                  'unseen_words':sum(unseen_count[s].values()),
                  'unseen_common_supported_words':sum(n for w,n in unseen_count[s].items() if eligible[s][w])} for s in vals}
    words={s:{'unseen':collections.Counter({w:n for w,n in unseen_count[s].items() if eligible[s][w]}),
              'all':actual[s]} for s in vals}
    for k,rr in enumerate(results):
        p=spm.SentencePieceProcessor(model_file=str(MODELS/(rr['name']+'.model')))
        rr['validation']={}
        for s in words:
            rr['validation'][s]={}
            for kind,cs in words[s].items():
                total=sum(n*alpha_len(w) for w,n in cs.items())
                pieces=sum(n*len(p.encode(w,out_type=int)) for w,n in cs.items())
                assert total>0
                rr['validation'][s][kind+'_E']=100*pieces/total
                rr['validation'][s][kind+'_alpha']=total
        rr['validation']['worst_unseen_E']=max(rr['validation'][s]['unseen_E'] for s in words)
        rr['validation']['worst_all_E']=max(rr['validation'][s]['all_E'] for s in words)
        if (k+1)%20==0:print('SCORED',k+1,'/',len(results),flush=True)
    return effective


def describe(results):
    rng=np.random.default_rng(20261023);report={};B=10000
    for mix in MIXES:
        by={method:np.array([r['validation']['worst_unseen_E'] for r in results if r['mix']==mix and r['method']==method]) for method in METHODS}
        assert all(len(a)==20 for a in by.values())
        group={}
        for method,arr in by.items():
            rows=[r for r in results if r['mix']==mix and r['method']==method]
            group[method]={'mean_worst_unseen_E':float(np.mean(arr)),
             'std_worst_unseen_E':float(np.std(arr,ddof=1)),
             'q90_worst_unseen_E':float(np.quantile(arr,.9)),
             'q10_worst_unseen_E':float(np.quantile(arr,.1)),
             'range10_90':float(np.quantile(arr,.9)-np.quantile(arr,.1)),
             'mean_IMVO_unseen_E':float(np.mean([r['validation']['IMVO']['unseen_E'] for r in rows])),
             'mean_ISIG_unseen_E':float(np.mean([r['validation']['ISIG']['unseen_E'] for r in rows])),
             'mean_IMVO_all_E':float(np.mean([r['validation']['IMVO']['all_E'] for r in rows])),
             'mean_ISIG_all_E':float(np.mean([r['validation']['ISIG']['all_E'] for r in rows])),
             'min_max':[float(min(arr)),float(max(arr))],
             'draw_values':arr.tolist()}
        base=by['word_uniform']
        group['comparisons']={}
        for method in METHODS[1:]:
            new=by[method]
            # Independently resampled bootstrap for cross-method membership draws.
            ind1=rng.integers(0,len(base),size=(B,len(base)))
            ind2=rng.integers(0,len(new),size=(B,len(new)))
            diffs=np.quantile(new[ind2],.9,axis=1)-np.quantile(base[ind1],.9,axis=1)
            mean_diff=np.mean(new[ind2],axis=1)-np.mean(base[ind1],axis=1)
            empirical=float(np.quantile(new,.9)-np.quantile(base,.9))
            emean=float(np.mean(new)-np.mean(base))
            interval=np.quantile(diffs,[.025,.975]).tolist()
            means=np.quantile(mean_diff,[.025,.975]).tolist()
            gate=bool(empirical<=-0.10 and emean<=0.10 and interval[1]<0)
            group['comparisons'][method+'_minus_word_uniform']={
              'q90_difference':empirical,'q90_bootstrap_95_ci':interval,
              'mean_difference':emean,'mean_bootstrap_95_ci':means,
              'practical_gate_pass':gate}
        report[f'imvo_mix_{int(mix*100)}']=group
    return report


def vocab_analysis(results):
    ret={}
    for mix in MIXES:
      for meth in METHODS:
        rows=[r for r in results if r['mix']==mix and r['method']==meth]
        vocabs=[]
        for row in rows:
            v=MODELS/(row['name']+'.vocab')
            vs={line.partition('\t')[0] for line in v.read_text().splitlines()}
            vocabs.append(vs)
        sims=[]
        for i in range(len(vocabs)):
            for j in range(i+1,len(vocabs)):
                sims.append(len(vocabs[i]&vocabs[j])/len(vocabs[i]|vocabs[j]))
        ret[f'{meth}_mix_{int(mix*100)}']={'mean_vocab_jaccard':statistics.mean(sims),
             'std_vocab_jaccard':statistics.stdev(sims),'pair_count':len(sims)}
    return ret


def baseline_replay_check():
    # Prove uniform word occurrence path returns exact saved model vocabulary at original member seed.
    p=training_pages()
    ww={s:[w for q in p if q['source']==s for w in forms(q['text'])] for s in ('IMVO','ISIG')}
    orig=sample_exact(ww['IMVO'],BUDGET,11*101+1)
    random.Random(11*101+3).shuffle(orig)
    prefix=ROOT/'replay_bpe_imvo_100_seed_11'
    with tempfile.TemporaryDirectory() as td:
        text=Path(td)/'corpus.txt'
        with text.open('w',encoding='utf8') as f:
            for i in range(0,len(orig),70):f.write(' '.join(orig[i:i+70])+'\n')
        spm.SentencePieceTrainer.train(input=str(text),model_prefix=str(prefix),
            model_type='bpe',vocab_size=2000,character_coverage=1.0,
            normalization_rule_name='identity',num_threads=2,hard_vocab_limit=False,
            bos_id=-1,eos_id=-1,pad_id=-1,unk_id=0,max_sentence_length=15000,
            shuffle_input_sentence=False,minloglevel=2)
    old=REF/'bpe_imvo_100_seed_11.vocab'
    assert prefix.with_suffix('.vocab').read_bytes()==old.read_bytes(),'prior model replay mismatch'
    print('PREVIOUS MODEL VOCAB REPRODUCED',flush=True)
    return hashlib.sha256(old.read_bytes()).hexdigest()

training_pages_cached=training_pages()

def main():
    sha=hashlib.sha256((ROOT/'PRECOMMIT.md').read_bytes()).hexdigest()
    assert sha=='78c1a1e3c538a62ab69fd4c3a7e612b7cee7546acd1b05566fcc1c8bf76235d1'
    mode=sys.argv[1] if len(sys.argv)>1 else 'train'
    checkpoint=ROOT/'training_checkpoint.json'
    all_params=[(mix,meth,seed) for mix in MIXES for meth in METHODS for seed in SEEDS]
    if mode=='train':
        start=int(sys.argv[2]) if len(sys.argv)>2 else 0
        stop=int(sys.argv[3]) if len(sys.argv)>3 else len(all_params)
        existing=json.loads(checkpoint.read_text())['runs'] if checkpoint.exists() else []
        assert start==len(existing),f'Expected start {len(existing)}, got {start}'
        data=issue_data()
        for i in range(start,stop):
            mix,meth,seed=all_params[i]
            row=train_one(data,meth,mix,seed)
            existing.append(row)
            checkpoint.write_text(json.dumps({'precommit_sha256':sha,'runs':existing},ensure_ascii=False,indent=2))
            print('TRAINED',i+1,'/',len(all_params),row['name'],round(time.monotonic()-START,2),'s',flush=True)
        print('CHUNK FINISHED',len(existing),'/',len(all_params),flush=True)
    elif mode=='score':
        existing=json.loads(checkpoint.read_text())['runs']
        assert len(existing)==len(all_params)
        assert [(r['mix'],r['method'],r['seed']) for r in existing]==all_params
        support=evaluate(existing,training_pages_cached)
        analyses=describe(existing)
        voc=vocab_analysis(existing)
        replay=hashlib.sha256((REF/'bpe_imvo_100_seed_11.vocab').read_bytes()).hexdigest()
        doc={'protocol':'PRECOMMITTED PHASE8 sampling stability, validation-retrospective only',
             'precommit_sha256':sha, 'replay_prior_vocab_sha256':replay,'mixtures':MIXES,
             'methods':METHODS,'seeds':SEEDS,'source_issues':{k:len(v) for k,v in issue_data().items()},
             'common_support':support,'analysis':analyses,'vocabulary_stability':voc,
             'runs':existing,'runtime_seconds_for_scoring':round(time.monotonic()-START,2)}
        OUT.write_text(json.dumps(doc,indent=2,ensure_ascii=False))
        for k,v in analyses.items():
            print(k)
            for meth in METHODS:print(' ',meth,{a:round(v[meth][a],4) for a in ('mean_worst_unseen_E','q90_worst_unseen_E','std_worst_unseen_E')})
            print('comparisons',v['comparisons'])
        print('FINISHED',OUT,round(time.monotonic()-START,2),'seconds',flush=True)
    elif mode=='replay':
        print('REPLAY',baseline_replay_check())
    else:raise ValueError(mode)

if __name__=='__main__':main()
