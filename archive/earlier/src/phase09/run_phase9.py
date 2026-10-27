#!/usr/bin/env python3
"""Phase 9 independent BPE train/score on frozen Phase8 matched word multisets."""
import collections, hashlib, json, os, random, sys, time
from pathlib import Path
import numpy as np
import sentencepiece as spm   # ONLY for reference score, not training alternative
from scipy.stats import spearmanr, pearsonr
from classic_bpe import ClassicBPE, run_unit_tests
ROOT=Path(__file__).resolve().parent
P8=Path('/mnt/data/isiXhosa_sampling_stability_phase8')
sys.path.insert(0,str(P8))
import run_phase8 as p8
assert hashlib.sha256((ROOT/'PRECOMMIT.md').read_bytes()).hexdigest()==(ROOT/'PRECOMMIT.sha256').read_text().split()[0]
assert hashlib.sha256((ROOT/'classic_bpe.py').read_bytes()).hexdigest()==(ROOT/'IMPLEMENTATION.sha256').read_text().split()[0]
assert p8.BUDGET==421696
SEEDS=list(range(1001,1021));MIXES=[.5,1.]
PARAMS=[(mix,seed) for mix in MIXES for seed in SEEDS]
MODELS=ROOT/'models';MODELS.mkdir(exist_ok=True)
CP=ROOT/'training_checkpoint.json'
allwords={src:[w for pg in p8.training_pages_cached if pg['source']==src for w in p8.forms(pg['text'])] for src in ('IMVO','ISIG')}


def get_sample(mix,seed):
    budget_i=round(p8.BUDGET*mix)
    sample=p8.sample_exact(allwords['IMVO'],budget_i,seed*101+1)
    sample+=p8.sample_exact(allwords['ISIG'],p8.BUDGET-budget_i,seed*101+2)
    assert sum(map(p8.alpha_len,sample))==p8.BUDGET
    random.Random(seed*101+3).shuffle(sample)
    return sample

def sample_sha(sample):
    # normalized serialized sample with explicit record boundaries
    return hashlib.sha256('\n'.join(sample).encode('utf-8')).hexdigest()

def replay_original():
    """The exact sample recreation is checked against Phase8 SP output vocabulary."""
    import tempfile
    s=get_sample(1.,1001)
    with tempfile.TemporaryDirectory() as td:
        inp=Path(td)/'corpus.txt'
        with inp.open('w',encoding='utf8') as f:
            for i in range(0,len(s),70): f.write(' '.join(s[i:i+70])+'\n')
        pre=Path(td)/'model'
        spm.SentencePieceTrainer.train(input=str(inp),model_prefix=str(pre),
            model_type='bpe',vocab_size=2000,character_coverage=1.0,
            normalization_rule_name='identity',num_threads=2,hard_vocab_limit=False,
            bos_id=-1,eos_id=-1,pad_id=-1,unk_id=0,max_sentence_length=15000,
            shuffle_input_sentence=False,minloglevel=2)
        original=p8.MODELS/'word_uniform_mix100_draw1001.vocab'
        assert pre.with_suffix('.vocab').read_bytes()==original.read_bytes(), 'sample recreation failure'
    return {'source_sample_sha256':sample_sha(s),'phase8_replay_vocab_identical':True}

def run_train(start,stop):
    rows=json.loads(CP.read_text())['runs'] if CP.exists() else []
    assert len(rows)==start,(len(rows),start)
    if start==0:
        (ROOT/'TOY_TESTS.json').write_text(json.dumps(run_unit_tests(),indent=2))
        (ROOT/'REPLAY.json').write_text(json.dumps(replay_original(),indent=2))
    for ix in range(start,stop):
        mix,seed=PARAMS[ix];t=time.monotonic()
        words=get_sample(mix,seed)
        m=ClassicBPE.fit(words,vocab_size=2000,min_frequency=2)
        name=f'classic_mix{int(mix*100):03d}_draw{seed}'
        dump=m.to_json()
        dump.update({'mix':mix,'seed':seed,'sample_sha256':sample_sha(words),
            'training_budget_alpha':p8.BUDGET,'sample_unique_types':len(set(words)),
            'word_occurrences':len(words)})
        (MODELS/(name+'.json')).write_text(json.dumps(dump,ensure_ascii=False))
        cs=collections.Counter(words)
        numer=sum(n*m.encode_len(w) for w,n in cs.items())
        denom=sum(n*p8.alpha_len(w) for w,n in cs.items())
        assert denom==p8.BUDGET
        row={'mix':mix,'seed':seed,'model_name':name,'model_sha256':hashlib.sha256((MODELS/(name+'.json')).read_bytes()).hexdigest(),
          'sample_sha256':dump['sample_sha256'],'word_occurrences':len(words),
          'sample_unique_types':len(cs),'effective_vocab':dump['effective_vocab'],
          'number_of_merges':len(m.merges),'self_train_E':100*numer/denom,
          'seconds':round(time.monotonic()-t,3)}
        rows.append(row)
        CP.write_text(json.dumps({'precommit_sha256':hashlib.sha256((ROOT/'PRECOMMIT.md').read_bytes()).hexdigest(),
                                  'implementation_sha256':hashlib.sha256((ROOT/'classic_bpe.py').read_bytes()).hexdigest(),
                                  'runs':rows},indent=2))
        print('CLASSIC_BPE',ix+1,'/',len(PARAMS),name,'merges',row['number_of_merges'],
            'vocab',row['effective_vocab'],'sec',row['seconds'],flush=True)
    print('CHUNK DONE',stop,'/',len(PARAMS),flush=True)


def score():
    from sklearn.mixture import GaussianMixture
    from scipy.stats import spearmanr,pearsonr
    rows=json.loads(CP.read_text())['runs'];assert len(rows)==40
    r8=json.loads(p8.OUT.read_text())['runs']
    spref={(x['mix'],x['seed']):x for x in r8 if x['method']=='word_uniform'}
    assert len(spref)==40
    models={}
    for row in rows:
        m=ClassicBPE.from_json(json.loads((MODELS/(row['model_name']+'.json')).read_text()))
        assert len(m.merges)==row['number_of_merges']
        models[(row['mix'],row['seed'])]=m
    seen={w for p in p8.training_pages_cached for w in p8.forms(p['text'])}
    vals={s:collections.Counter(w for p in p8.dev_pages() if p['source']==s for w in p8.forms(p['text'])) for s in ('IMVO','ISIG')}
    vocabs={s:list(vals[s].keys()) for s in vals}
    unseen={s:np.array([w not in seen for w in vocabs[s]],dtype=bool) for s in vals}
    alen={s:np.array([p8.alpha_len(w) for w in vocabs[s]],dtype=np.int64) for s in vals}
    weight={s:np.array([vals[s][w] for w in vocabs[s]],dtype=np.int64) for s in vals}
    valid={s:np.ones(len(vocabs[s]),dtype=bool) for s in vals}
    # SentencePiece word-encodability may be affected by rare characters.
    for i,row in enumerate(rows):
        m=models[(row['mix'],row['seed'])]
        sp=spm.SentencePieceProcessor(model_file=str(p8.MODELS/(f"word_uniform_mix{int(row['mix']*100):03d}_draw{row['seed']}.model")))
        for s in vals:
            valid[s]&=np.array([m.encode_len(w)!=-1 for w in vocabs[s]],dtype=bool)
            spouts=sp.encode(vocabs[s],out_type=int,num_threads=4)
            valid[s]&=np.array([sp.unk_id() not in z for z in spouts],dtype=bool)
        if (i+1)%10==0:print('COMMON_SUPPORT',i+1,'/',len(rows),flush=True)
    support={}
    for s in vals:
        support[s]={'valid_occurrences':int(weight[s][valid[s]].sum()),'all_occurrences':int(weight[s].sum()),
            'valid_unseen_occurrences':int(weight[s][valid[s]&unseen[s]].sum()),
            'unseen_occurrences':int(weight[s][unseen[s]].sum()),
            'valid_alpha':int((alen[s][valid[s]]*weight[s][valid[s]]).sum()),
            'valid_unseen_alpha':int((alen[s][valid[s]&unseen[s]]*weight[s][valid[s]&unseen[s]]).sum())}
        assert support[s]['valid_unseen_alpha']>0
    for i,row in enumerate(rows):
        key=(row['mix'],row['seed']);m=models[key]
        sp=spm.SentencePieceProcessor(model_file=str(p8.MODELS/(f"word_uniform_mix{int(row['mix']*100):03d}_draw{row['seed']}.model")))
        row['validation']={}
        for s in vals:
            counts=np.array([m.encode_len(w) for w in vocabs[s]],dtype=np.int32)
            spcounts=np.array([len(z) for z in sp.encode(vocabs[s],out_type=int,num_threads=4)],dtype=np.int32)
            row['validation'][s]={}
            for label,mask in [('unseen',valid[s]&unseen[s]),('all',valid[s])]:
                denom=int((alen[s][mask]*weight[s][mask]).sum())
                row['validation'][s][label+'_E']=100*int((counts[mask]*weight[s][mask]).sum())/denom
                row['validation'][s]['sp_'+label+'_E']=100*int((spcounts[mask]*weight[s][mask]).sum())/denom
        row['validation']['worst_unseen_E']=max(row['validation'][s]['unseen_E'] for s in vals)
        row['validation']['sp_worst_unseen_E']=max(row['validation'][s]['sp_unseen_E'] for s in vals)
        if (i+1)%10==0:print('SCORED',i+1,flush=True)
    summary={'protocol_sha256':hashlib.sha256((ROOT/'PRECOMMIT.md').read_bytes()).hexdigest(),
      'implementation_sha256':hashlib.sha256((ROOT/'classic_bpe.py').read_bytes()).hexdigest(),
      'source_sha256':hashlib.sha256(Path(p8.ORIGINAL_ZIP).read_bytes()).hexdigest() if hasattr(p8,'ORIGINAL_ZIP') else None,
      'support':support,'functions':{}}
    groups={mi:sorted([rr for rr in rows if rr['mix']==mi],key=lambda r:r['seed']) for mi in [0.5,1.]}
    for mi in [0.5,1.]:
        a=np.array([rr['validation']['worst_unseen_E'] for rr in groups[mi]])
        b=np.array([rr['validation']['sp_worst_unseen_E'] for rr in groups[mi]])
        comp=GaussianMixture(n_components=1,random_state=0,n_init=20,reg_covar=1e-6).fit(a.reshape(-1,1))
        comp2=GaussianMixture(n_components=2,random_state=0,n_init=20,reg_covar=1e-6).fit(a.reshape(-1,1))
        summary['functions'][f'mix_{int(mi*100)}']={'classic':{'mean':float(a.mean()),'std':float(a.std(ddof=1)),
             'min':float(a.min()),'max':float(a.max()),'values':a.tolist(),
             'bic_1_minus_bic_2':float(comp.bic(a.reshape(-1,1))-comp2.bic(a.reshape(-1,1))),
             'mixture_2_centers':sorted(comp2.means_.ravel().tolist())},
             'sentencepiece':{'mean':float(b.mean()),'std':float(b.std(ddof=1)),
             'values_same_support':b.tolist()},
             'matched_pearson':float(pearsonr(a,b).statistic),
             'matched_spearman':float(spearmanr(a,b).statistic)}
    one=groups[1.]
    a=np.array([rr['validation']['worst_unseen_E'] for rr in one])
    b=np.array([rr['validation']['sp_worst_unseen_E'] for rr in one])
    old=np.array([spref[1.,rr['seed']]['validation']['worst_unseen_E'] for rr in one])
    labels=old>=39.5
    obs_ratio=summary['functions']['mix_100']['classic']['std']/summary['functions']['mix_50']['classic']['std']
    diff=float(a[labels].mean()-a[~labels].mean())
    rs=np.random.default_rng(20261023);B=10000
    indices=rs.integers(0,20,size=(B,20))
    cors=[];deltas=[]
    for idx in indices:
        ca=a[idx];cb=b[idx];labs=labels[idx]
        cors.append(float(spearmanr(ca,cb).statistic) if len(set(ca))>1 and len(set(cb))>1 else float('nan'))
        deltas.append(float(ca[labs].mean()-ca[~labs].mean()) if 0<sum(labs)<20 else float('nan'))
    cors=np.array(cors);deltas=np.array(deltas)
    b50=np.array([rr['validation']['worst_unseen_E'] for rr in groups[.5]])
    halfindices=rs.integers(0,20,size=(B,20))
    sdratio=np.std(a[indices],axis=1,ddof=1)/np.std(b50[halfindices],axis=1,ddof=1)
    h1=bool(obs_ratio>=3.)
    spearman=float(spearmanr(a,b).statistic)
    ci_corr=np.nanquantile(cors,[.025,.975]).tolist()
    h2=bool(spearman>=.5 and ci_corr[0]>0)
    ci_diff=np.nanquantile(deltas,[.025,.975]).tolist()
    h3=bool(diff>=.4 and ci_diff[0]>0)
    h4=bool(summary['functions']['mix_100']['classic']['bic_1_minus_bic_2']>10 and summary['functions']['mix_50']['classic']['bic_1_minus_bic_2']<=5)
    summary['hypothesis_tests']={'H1_sensitivity':h1,'H2_matched_correlation':h2,'H3_old_SP_group_separation':h3,
          'H4_bimodality':h4,'primary_gate_H1_H2_H3':bool(h1 and h2 and h3),
          'classic_sd_ratio_100_over_50':float(obs_ratio),
          'classic_sd_ratio_bootstrap95':np.quantile(sdratio,[.025,.975]).tolist(),
          'matched_spearman':spearman,'matched_spearman_bootstrap95':ci_corr,
          'old_SP_high_group_n':int(sum(labels)),'old_SP_low_group_n':int(sum(~labels)),
          'classic_cost_high_minus_low':diff,
          'classic_cost_group_difference_bootstrap95':ci_diff}
    (ROOT/'results.json').write_text(json.dumps({'summary':summary,'runs':rows},ensure_ascii=False,indent=2))
    print('RESULTS',json.dumps(summary['hypothesis_tests'],indent=2),flush=True)
    print('GMM', {mi:summary['functions'][mi]['classic']['bic_1_minus_bic_2'] for mi in ('mix_100','mix_50')},flush=True)
    print('SAVED',ROOT/'results.json',flush=True)

if __name__=='__main__':
    arg=sys.argv[1]
    if arg=='train': run_train(int(sys.argv[2]),int(sys.argv[3]))
    elif arg=='score': score()
    else:raise ValueError('mode')
