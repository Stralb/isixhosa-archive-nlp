#!/usr/bin/env python3
"""Precommitted independent membership draws for training-only merge-branch rule."""
import collections,hashlib,json,sys,time,math
from pathlib import Path
import numpy as np
from classic_bpe_intervene import ClassicBPE
import run_phase9 as p
ROOT=p.ROOT
assert hashlib.sha256((ROOT/'FRESH_DRAW_RULE_PRECOMMIT.md').read_bytes()).hexdigest()==(ROOT/'FRESH_DRAW_RULE_PRECOMMIT.sha256').read_text().split()[0]
DEST=ROOT/'fresh_draws';DEST.mkdir(exist_ok=True)
CP=ROOT/'fresh_selection_checkpoint.json'
PARAMS=[(1.,seed) for seed in range(5001,5031)]+[(.5,seed) for seed in range(6001,6021)]
assert len(PARAMS)==50

def word_counts(words):
    cs=collections.Counter(words)
    paircounts=collections.Counter()
    for w,n in cs.items():
        t=['▁',*w]
        for pair,num in collections.Counter(zip(t,t[1:])).items():paircounts[pair]+=num*n
    top=sorted(paircounts.items(),key=lambda x:(-x[1],x[0]))[:2]
    return cs,top

def calculate_train_E(cs,model):
    return 100*sum(n*model.encode_len(w) for w,n in cs.items())/p.p8.BUDGET

def train(start,stop):
    rows=json.loads(CP.read_text())['runs'] if CP.exists() else []
    assert len(rows)==start
    for i in range(start,stop):
        mix,seed=PARAMS[i]
        sample=p.get_sample(mix,seed);assert sum(map(p.p8.alpha_len,sample))==p.p8.BUDGET
        cs,top=word_counts(sample)
        margin=(top[0][1]-top[1][1])/top[0][1]
        trigger=margin<=.02
        choices=[]
        for name,pair in [('greedy',top[0][0])]+([('second',top[1][0])] if trigger else []):
            t=time.monotonic()
            model=ClassicBPE.fit(sample,vocab_size=2000,forced_prefix=[pair])
            assert model.merges[0]==pair
            model_name=f'seed_{seed}_mix_{int(mix*100)}_{name}'
            payload=model.to_json()
            (DEST/(model_name+'.json')).write_text(json.dumps(payload,ensure_ascii=False))
            choices.append({'option':name,'pair':list(pair),'model_name':model_name,
                'model_sha256':hashlib.sha256((DEST/(model_name+'.json')).read_bytes()).hexdigest(),
                'training_E':100*model.train_token_count/p.p8.BUDGET,'fit_seconds':round(time.monotonic()-t,3)})
        selected=min(choices,key=lambda x:(x['training_E'],tuple(x['pair'])))
        # Save frozen *training-only* choice before any external model evaluation.
        row={'index':i,'source_mix':mix,'seed':seed,'char_budget':p.p8.BUDGET,
             'sample_sha256':p.sample_sha(sample),'training_word_occurrences':len(sample),
             'initial_top_pairs':[[list(pair),cnt] for pair,cnt in top],
             'initial_margin':margin,'trigger':trigger,'choices':choices,
             'selected_option':selected['option'],'selected_model':selected['model_name'],
             'selected_training_E':selected['training_E']}
        rows.append(row)
        CP.write_text(json.dumps({'sha256_precommit':hashlib.sha256((ROOT/'FRESH_DRAW_RULE_PRECOMMIT.md').read_bytes()).hexdigest(),
              'training_only':True,'runs':rows},indent=2,ensure_ascii=False))
        print('NEW_DRAW',i+1,'/50',mix,seed,'margin',round(margin,4),'trigger',trigger,
              'picked',selected['option'],'trainE',round(selected['training_E'],4),flush=True)
    print('CHUNK_DONE',stop,'/50',flush=True)

if __name__=='__main__':
    if sys.argv[1]=='train':train(int(sys.argv[2]),int(sys.argv[3]))
    else:raise ValueError(sys.argv[1])
