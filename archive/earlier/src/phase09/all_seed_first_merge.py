#!/usr/bin/env python3
"""Phase9C: causal paired first-merge intervention, resumable by sample seed."""
import collections,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from classic_bpe_intervene import ClassicBPE
import run_phase9 as p
ROOT=p.ROOT; P8=p.p8
assert hashlib.sha256((ROOT/'ALL_SEED_FIRST_MERGE_PRECOMMIT.md').read_bytes()).hexdigest()==(ROOT/'ALL_SEED_FIRST_MERGE_PRECOMMIT.sha256').read_text().split()[0]
PIVOT_A=('n','g');PIVOT_B=('▁','n')
rows=json.loads((ROOT/'results.json').read_text())['runs'];ref={r['seed']:r for r in rows if r['mix']==1.}
CACHED=ROOT/'encodings'
TRAIN={w for pg in P8.training_pages_cached for w in P8.forms(pg['text'])}
raw={d:collections.Counter(w for pg in P8.dev_pages() if pg['source']==d for w in P8.forms(pg['text'])) for d in ('IMVO','ISIG')}
lex={d:list(raw[d]) for d in raw}
weights={d:np.array([raw[d][w] for w in lex[d]],dtype=np.int64) for d in raw}
lens={d:np.array([P8.alpha_len(w) for w in lex[d]],dtype=np.int64) for d in raw}
novel={d:np.array([w not in TRAIN for w in lex[d]],dtype=bool) for d in raw}
cache=[]
for r in rows:
    with np.load(CACHED/(r['model_name']+'.npz')) as z:
        cache.append({k:z[k] for k in z.files})
valid={d:np.logical_and.reduce([z[d+'_classic']!=-1 for z in cache]+[z[d+'_sp']!=-1 for z in cache]) for d in raw}
for d in raw:
    assert (weights[d][novel[d]&valid[d]]*lens[d][novel[d]&valid[d]]).sum()==json.loads((ROOT/'results.json').read_text())['summary']['support'][d]['valid_unseen_alpha']
results_path=ROOT/'first_merge_all_seed_results.json'
MODELS=ROOT/'first_merge_models';MODELS.mkdir(exist_ok=True)

def score(model):
    ret={}
    for d in raw:
        mask=valid[d]&novel[d]
        x=np.array([model.encode_len(w) for w,keep in zip(lex[d],mask) if keep],dtype=np.int32)
        assert (x>0).all()
        ret[d]=100*int((x*weights[d][mask]).sum())/int((lens[d][mask]*weights[d][mask]).sum())
    return ret,max(ret.values())


def train(start,stop):
    records=json.loads(results_path.read_text())['records'] if results_path.exists() else []
    assert len(records)==2*start,len(records)
    for i in range(start,stop):
        seed=1001+i
        sampled=p.get_sample(1.,seed)
        assert p.sample_sha(sampled)==ref[seed]['sample_sha256']
        original=ClassicBPE.from_json(json.loads((p.MODELS/(ref[seed]['model_name']+'.json')).read_text()))
        for treatment,pivot in [('ng',PIVOT_A),('word_start_n',PIVOT_B)]:
            st=time.monotonic();name=f'draw{seed}_first_{treatment}'
            model=ClassicBPE.fit(sampled,vocab_size=2000,forced_prefix=[pivot])
            assert model.merges[0]==pivot
            if original.merges[0]==pivot:
                assert model.merges==original.merges,('replay failed',seed,treatment)
            vals,maximum=score(model)
            (MODELS/(name+'.json')).write_text(json.dumps(model.to_json(),ensure_ascii=False))
            records.append({'seed':seed,'treatment':treatment,'first_merge':list(pivot),
                'natural_first_merge':list(original.merges[0]),'is_original_first_merge':original.merges[0]==pivot,
                'sample_sha256':ref[seed]['sample_sha256'],'worst_unseen_E':maximum,'per_domain_unseen_E':vals,
                'train_time_and_score_seconds':time.monotonic()-st,'n_merges':len(model.merges)})
            results_path.write_text(json.dumps({'protocol_sha256':hashlib.sha256((ROOT/'ALL_SEED_FIRST_MERGE_PRECOMMIT.md').read_bytes()).hexdigest(),
                'records':records},indent=2,ensure_ascii=False))
            print('FIRST_MERGE',seed,treatment,'natural',original.merges[0],
                'worstE',round(maximum,4),'seconds',round(time.monotonic()-st,2),flush=True)
    print('CHUNK DONE',len(records),'/',40)


def summarize():
    records=json.loads(results_path.read_text())['records'];assert len(records)==40
    m={(r['seed'],r['treatment']):r for r in records}
    paired=[]
    for seed in p.SEEDS:
        a=m[seed,'ng'];b=m[seed,'word_start_n'];old=ref[seed]
        worst0=old['validation']['worst_unseen_E']
        matched=a if a['is_original_first_merge'] else b if b['is_original_first_merge'] else None
        if matched is not None: assert abs(matched['worst_unseen_E']-worst0)<1e-7
        paired.append({'seed':seed,'ng_E':a['worst_unseen_E'],'word_start_n_E':b['worst_unseen_E'],
             'delta_wordstart_minus_ng':b['worst_unseen_E']-a['worst_unseen_E'],
             'natural_first_merge':a['natural_first_merge'],
             'natural_E':worst0,'ng_improvement':worst0-a['worst_unseen_E'],
             'natural_replayed':matched is not None})
    arr=np.array([r['delta_wordstart_minus_ng'] for r in paired])
    rng=np.random.default_rng(20261023);B=10000;idx=rng.integers(0,20,size=(B,20))
    ci=np.quantile(arr[idx].mean(axis=1),[.025,.975]).tolist()
    npos=int(sum(arr>0))
    summary={'mean_delta':float(arr.mean()),'median_delta':float(np.median(arr)),
      'min_delta':float(arr.min()),'max_delta':float(arr.max()),
      'positive_seeds':npos,'paired_bootstrap95':ci,
      'precommitted_gate_pass':bool(arr.mean()>=.7 and ci[0]>0 and npos>=18),
      'natural_first_merge_counts':{' + '.join(k):v for k,v in collections.Counter(tuple(r['natural_first_merge']) for r in paired).items()},
      'ng_worst_E_mean':float(np.mean([r['ng_E'] for r in paired])),
      'ng_worst_E_sd':float(np.std([r['ng_E'] for r in paired],ddof=1)),
      'wordstart_E_mean':float(np.mean([r['word_start_n_E'] for r in paired])),
      'wordstart_E_sd':float(np.std([r['word_start_n_E'] for r in paired],ddof=1)),
      'natural_E_mean':float(np.mean([r['natural_E'] for r in paired])),
      'natural_E_sd':float(np.std([r['natural_E'] for r in paired],ddof=1))}
    doc=json.loads(results_path.read_text());doc['paired']=paired;doc['summary']=summary
    results_path.write_text(json.dumps(doc,indent=2,ensure_ascii=False))
    print('SUMMARY',json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='train':train(int(sys.argv[2]),int(sys.argv[3]))
    elif sys.argv[1]=='summary':summarize()
    else:raise ValueError('command')
