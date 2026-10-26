#!/usr/bin/env python3
import collections,json,time
from pathlib import Path
import numpy as np
import sentencepiece as spm
import run_phase8 as r
start=time.monotonic()
new=json.load(open(r.ROOT/'unigram_training_checkpoint.json'))['runs']
old=[x for x in json.load(open(r.ROOT/'training_checkpoint.json'))['runs'] if x['method']=='word_uniform']
assert len(new)==40 and len(old)==40
seeds={}
for z in new:seeds['unigram',z['mix'],z['seed']]=z['model_file']
for z in old:seeds['bpe',z['mix'],z['seed']]=str(r.MODELS/(z['name']+'.model'))
seen={w for p in r.training_pages_cached for w in r.forms(p['text'])}
raw={s:collections.Counter(w for p in r.dev_pages() if p['source']==s for w in r.forms(p['text'])) for s in ('IMVO','ISIG')}
words={s:list(raw[s]) for s in raw}
n={s:np.array([raw[s][w] for w in words[s]],dtype=int) for s in raw}
cs={s:np.array([r.alpha_len(w) for w in words[s]],dtype=int) for s in raw}
novel={s:np.array([w not in seen for w in words[s]],dtype=bool) for s in raw}
alphabet={c for p in r.training_pages_cached for w in r.forms(p['text']) for c in w if c.isalpha()}
valid={s:np.array([all(not c.isalpha() or c in alphabet for c in w) for w in words[s]],dtype=bool) for s in raw}
keys=sorted(seeds,key=lambda x:(x[1],x[2],x[0]))
enc=[]
for (alg,mix,seed) in keys:
    p=spm.SentencePieceProcessor(model_file=seeds[(alg,mix,seed)])
    qq={}
    for s in raw:
       cc=p.encode(words[s],out_type=int,num_threads=4)
       qq[s]=np.array([len(c) if p.unk_id() not in c else -1 for c in cc],dtype=int)
       valid[s]&=(qq[s]>=0)
    enc.append(qq)
results=[]
for key,cnt in zip(keys,enc):
    item={'algorithm':key[0],'mix':key[1],'seed':key[2],'validation':{}}
    for s in raw:
        item['validation'][s]={}
        for typ,mask in [('unseen',valid[s]&novel[s]),('all',valid[s])]:
            item['validation'][s][typ+'_E']=float(100*np.sum(n[s][mask]*cnt[s][mask])/np.sum(n[s][mask]*cs[s][mask]))
    item['validation']['worst_unseen_E']=max(item['validation'][s]['unseen_E'] for s in raw)
    results.append(item)

B=10000;rng=np.random.default_rng(20261027);report={}
for mix in (.5,1.):
    by={alg:np.array([z['validation']['worst_unseen_E'] for z in results if z['algorithm']==alg and z['mix']==mix]) for alg in ('bpe','unigram')}
    assert all(len(q)==20 for q in by.values())
    idx=rng.integers(0,20,size=(B,20))
    b=by['bpe'];u=by['unigram']
    bootratio=np.std(u[idx],ddof=1,axis=1)/np.std(b[idx],ddof=1,axis=1)
    ci=np.quantile(bootratio,[.025,.975]).tolist()
    report[str(mix)]={}
    for alg,x in by.items():
       report[str(mix)][alg]={'mean':float(np.mean(x)),'std':float(np.std(x,ddof=1)),
         'q90':float(np.quantile(x,.9)),'q10':float(np.quantile(x,.1)),
         'mean_IMVO_unseen_E':float(np.mean([z['validation']['IMVO']['unseen_E'] for z in results if z['algorithm']==alg and z['mix']==mix])),
         'mean_ISIG_unseen_E':float(np.mean([z['validation']['ISIG']['unseen_E'] for z in results if z['algorithm']==alg and z['mix']==mix])),
         'values':x.tolist()}
    ratio=float(np.std(u,ddof=1)/np.std(b,ddof=1))
    report[str(mix)]['unigram_vs_bpe']={'sd_ratio':ratio,'sd_ratio_bootstrap95':ci,
       'mean_difference':float(np.mean(u)-np.mean(b)),
       'q90_difference':float(np.quantile(u,.9)-np.quantile(b,.9)),
       'diagnostic_gate':bool(ratio<.5 and ci[1]<1 and np.mean(u)<=np.mean(b)+.20)}

voc_stability={}
for mix in (.5,1.):
 for alg in ('bpe','unigram'):
    vs=[]
    for seed in r.SEEDS:
      file=Path(seeds[(alg,mix,seed)]).with_suffix('.vocab')
      vs.append({line.partition('\t')[0] for line in file.read_text().splitlines()})
    sim=[len(vs[i]&vs[j])/len(vs[i]|vs[j]) for i in range(20) for j in range(i+1,20)]
    voc_stability[f'{mix}_{alg}']={'mean_jaccard':float(np.mean(sim)),'std_jaccard':float(np.std(sim,ddof=1))}

support={s:{'total_words':int(np.sum(n[s])),'supported_words':int(np.sum(n[s][valid[s]])),
            'unseen_words':int(np.sum(n[s][novel[s]])),
            'supported_unseen_words':int(np.sum(n[s][novel[s]&valid[s]]))} for s in raw}
output={'report':report,'common_support':support,'vocabulary_stability':voc_stability,
        'deterministic_unigram_refit':json.load(open(r.ROOT/'unigram_determinism.json'))['byte_identical_vocab'],
        'runs':results,'runtime_s':time.monotonic()-start,
        'scope':'Precommitted retrospective diagnostic, same sampled corpus membership and word order.'}
f=r.ROOT/'unigram_results.json';f.write_text(json.dumps(output,indent=2))
for m,v in report.items():
    print('MIX',m)
    for alg in ('bpe','unigram'):print(alg,{k:round(v[alg][k],4) for k in ('mean','std','q90')})
    print('DIFF',v['unigram_vs_bpe'])
print('SUPPORT',support,'DETERMINISM',output['deterministic_unigram_refit'])
print('FINISHED',f,round(time.monotonic()-start,2))
