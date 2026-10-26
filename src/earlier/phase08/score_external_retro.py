#!/usr/bin/env python3
"""Retrospective transfer of frozen Phase8C train-only selector onto 3 existing external corpora."""
import collections,hashlib,json,time,sys
from pathlib import Path
import numpy as np
import sentencepiece as spm
import run_phase8 as r
st=time.monotonic()
root=r.ROOT
MNT=Path('/mnt/data')
phase=json.load(open(root/'selfscore_results.json'))
new=phase['new_models']
choices=[min(new[3*i:3*i+3],key=lambda z:(z['self_train_E'],z['seed'])) for i in range(20)]
first=[new[3*i] for i in range(20)]
reference=[z for z in phase['baseline_models'] if z['mix']==.5]
triple=json.load(open(root/'triplemass_training_checkpoint.json'))['runs']
full=json.load(open(root/'large_single_training.json'))['runs'][1]
assert all(len(c)==20 for c in (choices,first,reference,triple))
model_spec=[]
for category,rows in [('selected',choices),('first',first),('balanced',reference)]:
    for z in rows:model_spec.append((category,z['seed'],str(r.MODELS/(z['name']+'.model'))))
for z in triple:model_spec.append(('large_single',z['seed'],z['model_file']))
model_spec.append(('full_pool',0,full['model_file']))
assert len(model_spec)==81
train_seen={w for p in r.training_pages_cached for w in r.forms(p['text'])}
train_alpha={c for w in train_seen for c in w if c.isalpha()}

import json as _json
corpora={}
for label,filename in [('ityala','ityala_lamawele_raw.txt'),('zem','zemkinkomo_magwalandini_raw.txt')]:
    text=(MNT/'isiXhosa_historical_eval/ocr'/filename).read_text(encoding='utf8')
    corpora[label]=[{'id':label,'text':text}]
modern=[]
for i,line in enumerate((MNT/'xho.jsonl').read_text(encoding='utf8').splitlines()):
    if not line.strip():continue
    z=_json.loads(line)
    t=z.get('text','')
    if not isinstance(t,str) or r.alpha_len(t)==0:continue
    modern.append({'id':str(i),'text':t})
corpora['modern_vukuzenzele']=modern
assert len(modern)==181
out={}
for label,docs in corpora.items():
    allforms=collections.Counter(w for x in docs for w in r.forms(x['text']))
    unique=list(allforms)
    weights=np.array([allforms[w] for w in unique],dtype=np.int64)
    alen=np.array([r.alpha_len(w) for w in unique],dtype=np.int64)
    unseen=np.array([w not in train_seen for w in unique],dtype=bool)
    allowed=np.array([all(not c.isalpha() or c in train_alpha for c in w) for w in unique],dtype=bool)
    enc=[]
    for j,(category,seed,path) in enumerate(model_spec):
        sp=spm.SentencePieceProcessor(model_file=path)
        vec=sp.encode(unique,out_type=int,num_threads=4)
        zz=np.array([len(v) if sp.unk_id() not in v else -1 for v in vec],dtype=np.int32)
        allowed&=(zz!=-1)
        enc.append(zz)
        if (j+1)%20==0:print(label,'ENCODED',j+1,'/',len(model_spec),'time',round(time.monotonic()-st,1),flush=True)
    novel=unseen&allowed
    fullmask=allowed
    assert novel.sum()>0
    stats={'n_documents':len(docs),'total_forms':int(sum(weights)),
           'unseen_forms':int(sum(weights[unseen])),
           'common_supported_forms':int(sum(weights[allowed])),
           'common_supported_unseen_forms':int(sum(weights[novel])),
           'common_unseen_alpha_chars':int(sum(weights[novel]*alen[novel])),
           'per_policy':{},'raw_scores':[]}
    for (policy,seed,path),ev in zip(model_spec,enc):
       score={'policy':policy,'seed':seed}
       for typ,mask in [('unseen',novel),('all',fullmask)]:
          numerator=int(np.sum(weights[mask]*ev[mask]));denom=int(np.sum(weights[mask]*alen[mask]));score[typ+'_E']=100*numerator/denom
       stats['raw_scores'].append(score)
    B=10000;rng=np.random.default_rng(20261028)
    for policy in ['first','selected','balanced','large_single','full_pool']:
        v=np.array([z['unseen_E'] for z in stats['raw_scores'] if z['policy']==policy])
        a=np.array([z['all_E'] for z in stats['raw_scores'] if z['policy']==policy])
        stats['per_policy'][policy]={'n':len(v),'unseen_mean':float(v.mean()),'unseen_q90':float(np.quantile(v,.9)) if len(v)>1 else None,
              'unseen_sd':float(np.std(v,ddof=1)) if len(v)>1 else None,
              'all_mean':float(a.mean()),
              'unseen_values':v.tolist()}
    sels=np.array(stats['per_policy']['selected']['unseen_values'])
    first=np.array(stats['per_policy']['first']['unseen_values'])
    ids=rng.integers(0,20,size=(B,20))
    diffs=np.mean(sels[ids]-first[ids],axis=1)
    stats['selected_minus_first']={'mean':float(np.mean(sels-first)),
         'paired_bootstrap95':np.quantile(diffs,[.025,.975]).tolist(),
         'selected_wins_triplets':int(np.sum(sels<first))}
    out[label]=stats

report={'protocol':'EXTERNAL_RETRO_PROTOCOL.md',
        'source_note':'Retrospective externally inspected corpora. Current policies fixed using newspaper training/retrospective validation only.',
        'datasets':out,'runtime_s':time.monotonic()-st}
path=root/'external_retro_results.json';path.write_text(json.dumps(report,indent=2))
for label,data in out.items():
    print('CORPUS',label,'forms',data['total_forms'],'support novel',data['common_supported_unseen_forms'])
    for category,rr in data['per_policy'].items():print(category,round(rr['unseen_mean'],4), 'q90',rr['unseen_q90'],'all',round(rr['all_mean'],4))
    print('selected-first',data['selected_minus_first'])
print('DONE',path,'total time',round(time.monotonic()-st,2))
