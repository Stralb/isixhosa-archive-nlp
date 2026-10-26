#!/usr/bin/env python3
"""Equivalent vector-batched scoring; same fixed common support as precommit.
Replaces slow nested scalar SentencePiece calls, without changing treatments or outcome.
"""
import collections,hashlib,json,time
from pathlib import Path
import numpy as np
import sentencepiece as spm
import run_phase8 as r
st=time.monotonic()
precommit=hashlib.sha256((r.ROOT/'PRECOMMIT.md').read_bytes()).hexdigest()
assert precommit=='78c1a1e3c538a62ab69fd4c3a7e612b7cee7546acd1b05566fcc1c8bf76235d1'
rows=json.loads((r.ROOT/'training_checkpoint.json').read_text())['runs']
assert len(rows)==120
trained=r.training_pages_cached
allseen={w for p in trained for w in r.forms(p['text'])}
raw={s:collections.Counter(w for p in r.dev_pages() if p['source']==s for w in r.forms(p['text'])) for s in ('IMVO','ISIG')}
lex={s:list(raw[s]) for s in raw}
weights={s:np.array([raw[s][w] for w in lex[s]],dtype=np.int64) for s in raw}
alpha={s:np.array([r.alpha_len(w) for w in lex[s]],dtype=np.int64) for s in raw}
unseen={s:np.array([w not in allseen for w in lex[s]],dtype=bool) for s in raw}
alphabet={c for p in trained for w in r.forms(p['text']) for c in w if c.isalpha()}
allowed={s:np.array([all(not c.isalpha() or c in alphabet for c in w) for w in lex[s]],dtype=bool) for s in raw}
encodings=[]
for i,entry in enumerate(rows):
    p=spm.SentencePieceProcessor(model_file=str(r.MODELS/(entry['name']+'.model')))
    result={}
    for s in ('IMVO','ISIG'):
        encoded=p.encode(lex[s],out_type=int,num_threads=4)
        result[s]=np.array([len(ids) if p.unk_id() not in ids else -1 for ids in encoded],dtype=np.int32)
        allowed[s]&=(result[s]!=-1)
    encodings.append(result)
    if (i+1)%30==0:print('ENCODED',i+1,'/',len(rows),round(time.monotonic()-st,2),flush=True)

support={}
for s in ('IMVO','ISIG'):
    selected=allowed[s]
    support[s]={'total_words':int(sum(weights[s])),
           'valid_words':int(sum(weights[s][selected])),
           'total_alpha':int(np.sum(weights[s]*alpha[s])),
           'valid_alpha':int(np.sum(weights[s][selected]*alpha[s][selected])),
           'unseen_words':int(sum(weights[s][unseen[s]])),
           'unseen_common_supported_words':int(sum(weights[s][unseen[s]&selected]))}
for i,row in enumerate(rows):
    row['validation']={}
    for s in ('IMVO','ISIG'):
        row['validation'][s]={}
        for k,sel in (('all',allowed[s]),('unseen',allowed[s]&unseen[s])):
            denom=int(sum(weights[s][sel]*alpha[s][sel]))
            numer=int(sum(weights[s][sel]*encodings[i][s][sel]))
            assert denom>0 and (encodings[i][s][sel]>=0).all()
            row['validation'][s][k+'_E']=100*numer/denom
            row['validation'][s][k+'_alpha']=denom
    row['validation']['worst_unseen_E']=max(row['validation'][s]['unseen_E'] for s in ('IMVO','ISIG'))
    row['validation']['worst_all_E']=max(row['validation'][s]['all_E'] for s in ('IMVO','ISIG'))

analysis=r.describe(rows)
voc=r.vocab_analysis(rows)
replay=hashlib.sha256((r.REF/'bpe_imvo_100_seed_11.vocab').read_bytes()).hexdigest()
doc={'protocol':'PRECOMMITTED PHASE8 sampling stability, validation-retrospective only',
   'precommit_sha256':precommit,'replay_prior_vocab_sha256':replay,'mixtures':r.MIXES,
   'methods':r.METHODS,'seeds':r.SEEDS,'source_issues':{k:len(v) for k,v in r.issue_data().items()},
   'common_support':support,'analysis':analysis,'vocabulary_stability':voc,
   'runs':rows,'runtime_seconds_for_scoring':round(time.monotonic()-st,2),
   'implementation_note':'Batched SentencePiece word scoring; same encodability intersection across all 120 fitted models.'}
r.OUT.write_text(json.dumps(doc,indent=2,ensure_ascii=False))
for key,item in analysis.items():
    print('MIX',key)
    for method in r.METHODS:
        v=item[method]
        print(' ',method, {k:round(v[k],4) for k in ('mean_worst_unseen_E','q90_worst_unseen_E','std_worst_unseen_E','mean_IMVO_unseen_E','mean_ISIG_unseen_E')})
    print(' comparisons',item['comparisons'])
print('DONE',r.OUT,'elapsed',round(time.monotonic()-st,2),flush=True)
