#!/usr/bin/env python3
"""Retrospective dose-response scoring on an identical common support across 100 models."""
import collections,hashlib,json,time
from pathlib import Path
import numpy as np
import sentencepiece as spm
import run_phase8 as r

st=time.monotonic()
pr=hashlib.sha256((r.ROOT/'SOURCE_MIX_PRECOMMIT.md').read_bytes()).hexdigest()
base=json.loads((r.ROOT/'training_checkpoint.json').read_text())['runs']
base=[z for z in base if z['method']=='word_uniform' and z['mix'] in (.5,1.0)]
new=json.loads((r.ROOT/'source_mix_checkpoint.json').read_text())['runs']
assert len(base)==40 and len(new)==60
rows=sorted(base+new,key=lambda d:(d['mix'],d['seed']))
MIXES=[.5,.75,.9,.95,1.0]
for mix in MIXES:assert len([x for x in rows if x['mix']==mix])==20
trained=r.training_pages_cached
seen={w for p in trained for w in r.forms(p['text'])}
raw={s:collections.Counter(w for p in r.dev_pages() if p['source']==s for w in r.forms(p['text'])) for s in ('IMVO','ISIG')}
lex={s:list(raw[s]) for s in raw}
weights={s:np.array([raw[s][w] for w in lex[s]],dtype=np.int64) for s in raw}
alpha={s:np.array([r.alpha_len(w) for w in lex[s]],dtype=np.int64) for s in raw}
unseen={s:np.array([w not in seen for w in lex[s]],dtype=bool) for s in raw}
alphabet={c for p in trained for w in r.forms(p['text']) for c in w if c.isalpha()}
valid={s:np.array([all(not c.isalpha() or c in alphabet for c in w) for w in lex[s]],dtype=bool) for s in raw}
enc=[]
for i,row in enumerate(rows):
    p=spm.SentencePieceProcessor(model_file=str(r.MODELS/(row['name']+'.model')))
    this={}
    for s in raw:
        seq=p.encode(lex[s],out_type=int,num_threads=4)
        this[s]=np.array([len(c) if p.unk_id() not in c else -1 for c in seq],dtype=np.int32)
        valid[s]&=(this[s]!=-1)
    enc.append(this)
    if (i+1)%20==0:print('ENCODED',i+1,'/',len(rows),round(time.monotonic()-st,2),flush=True)

support={}
for s in raw:
    support[s]={'total_words':int(sum(weights[s])),'supported_words':int(sum(weights[s][valid[s]])),
      'unseen_words':int(sum(weights[s][unseen[s]])),
      'supported_unseen_words':int(sum(weights[s][valid[s]&unseen[s]]))}
for i,row in enumerate(rows):
  row['validation']={}
  for s in raw:
    row['validation'][s]={}
    for typ,mask in [('all',valid[s]),('unseen',valid[s]&unseen[s])]:
        num=int(sum(weights[s][mask]*enc[i][s][mask]))
        den=int(sum(weights[s][mask]*alpha[s][mask]))
        assert den>0
        row['validation'][s][typ+'_E']=100*num/den
        row['validation'][s][typ+'_alpha']=den
  row['validation']['worst_unseen_E']=max(row['validation'][s]['unseen_E'] for s in raw)
  row['validation']['worst_all_E']=max(row['validation'][s]['all_E'] for s in raw)

rng=np.random.default_rng(20261024); B=10000
b=np.array([row['validation']['worst_unseen_E'] for row in rows if row['mix']==1.0])
report={}; arrays={}
for mix in MIXES:
  sub=[x for x in rows if x['mix']==mix]
  vals=np.array([x['validation']['worst_unseen_E'] for x in sub]);arrays[mix]=vals
  boot1=rng.integers(0,len(vals),(B,len(vals)));boot2=rng.integers(0,len(b),(B,len(b)))
  s1=np.std(vals[boot1],ddof=1,axis=1);s2=np.std(b[boot2],ddof=1,axis=1)
  ratio=np.divide(s1,s2,out=np.full_like(s1,np.nan),where=s2>0)
  delta_q90=np.quantile(vals[boot1],.9,axis=1)-np.quantile(b[boot2],.9,axis=1)
  sd=float(np.std(vals,ddof=1)); base_sd=float(np.std(b,ddof=1)); ratio_real=sd/base_sd
  q90_diff=float(np.quantile(vals,.9)-np.quantile(b,.9))
  ci_ratio=np.nanquantile(ratio,[.025,.975]).tolist();ci_q90=np.quantile(delta_q90,[.025,.975]).tolist()
  report[str(mix)]={'mean':float(np.mean(vals)),'median':float(np.median(vals)),
       'std':sd,'q10':float(np.quantile(vals,.1)),'q90':float(np.quantile(vals,.9)),
       'range10_90':float(np.quantile(vals,.9)-np.quantile(vals,.1)),
       'min':float(np.min(vals)),'max':float(np.max(vals)),
       'mean_IMVO_unseen_E':float(np.mean([x['validation']['IMVO']['unseen_E'] for x in sub])),
       'mean_ISIG_unseen_E':float(np.mean([x['validation']['ISIG']['unseen_E'] for x in sub])),
       'mean_IMVO_all_E':float(np.mean([x['validation']['IMVO']['all_E'] for x in sub])),
       'mean_ISIG_all_E':float(np.mean([x['validation']['ISIG']['all_E'] for x in sub])),
       'sd_ratio_vs_100':float(ratio_real),'sd_ratio_bootstrap_95_ci':ci_ratio,
       'q90_diff_vs_100':q90_diff,'q90_diff_bootstrap_95_ci':ci_q90,
       'stabilization_exploratory_gate':bool(mix<1 and ratio_real<=.5 and q90_diff<=-.20 and ci_ratio[1]<1),
       'values':vals.tolist()}

# Vocab stability by mix, around n=20 draw pairs.
vs={}
for mix in MIXES:
    sub=[x for x in rows if x['mix']==mix];tmp=[]
    for z in sub:
        words={line.partition('\t')[0] for line in (r.MODELS/(z['name']+'.vocab')).read_text().splitlines()}
        tmp.append(words)
    jac=[len(tmp[i]&tmp[j])/len(tmp[i]|tmp[j]) for i in range(20) for j in range(i+1,20)]
    vs[str(mix)]={'mean':float(np.mean(jac)),'sd':float(np.std(jac,ddof=1)),'pair_count':len(jac)}

doc={'protocol':'Phase8B precommitted exploratory source mix intervention on retrospective 1887 newspaper validation',
      'precommit_sha256':pr,'common_support':support,'mixes':MIXES,
      'results_by_mix':report,'vocab_stability':vs,'runs':rows,
      'evaluation_runtime_sec':round(time.monotonic()-st,2)}
p=r.ROOT/'source_mix_results.json';p.write_text(json.dumps(doc,indent=2))
for mix in MIXES:
   z=report[str(mix)]
   print('MIX',mix,'MEAN',round(z['mean'],4),'SD',round(z['std'],4),
       'Q90',round(z['q90'],4),'SD_ratio',round(z['sd_ratio_vs_100'],3),'SD_ratio_ci',z['sd_ratio_bootstrap_95_ci'],
       'Q90_diff',round(z['q90_diff_vs_100'],4),'Q90_ci',z['q90_diff_bootstrap_95_ci'],
       'GATE',z['stabilization_exploratory_gate'],flush=True)
print('FINISHED',p,'time',round(time.monotonic()-st,2),flush=True)
