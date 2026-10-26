#!/usr/bin/env python3
import collections,json,statistics,sys
from pathlib import Path
import numpy as np
import sentencepiece as spm
import run_phase8 as r
root=r.ROOT
base=json.loads((root/'source_mix_results.json').read_text())
models=[z for z in base['runs'] if z['mix']==1.0]
assert len(models)==20

def cost(processor,counts):
    items=list(counts.items());enc=processor.encode([w for w,n in items],out_type=int,num_threads=4)
    num=sum(n*len(ids) for (w,n),ids in zip(items,enc))
    denom=sum(n*r.alpha_len(w) for w,n in items)
    return 100*num/denom

traincnt=collections.Counter(w for p in r.training_pages_cached if p['source']=='IMVO' for w in r.forms(p['text']))
seen={w for p in r.training_pages_cached for w in r.forms(p['text'])}
vals={s:collections.Counter(w for p in r.dev_pages() if p['source']==s for w in r.forms(p['text'])) for s in ('IMVO','ISIG')}
val_seen={s:collections.Counter({w:n for w,n in vals[s].items() if w in seen}) for s in vals}
val_novel={s:collections.Counter({w:n for w,n in vals[s].items() if w not in seen}) for s in vals}

arr=[];vocabs={}
for z in models:
    name=z['name'];sp=spm.SentencePieceProcessor(model_file=str(r.MODELS/(name+'.model')))
    pieces=[p for p in sp.id_to_piece(list(range(sp.get_piece_size())))]
    vocab=set(pieces)
    vocabs[name]=vocab
    has_space=[x for x in pieces if x.startswith('▁')]
    whole_words=[w for w in has_space if len(w)>4]
    alpha_count=sum(len(w) for w in pieces)
    summary={'name':name,'seed':z['seed'],
          'band':'low' if z['validation']['worst_unseen_E']<39.5 else 'high',
          'worst_unseen_E':z['validation']['worst_unseen_E'],
          'imvo_val_unseen_E':z['validation']['IMVO']['unseen_E'],
          'isig_val_unseen_E':z['validation']['ISIG']['unseen_E'],
          'imvo_val_all_E':z['validation']['IMVO']['all_E'],
          'isig_val_all_E':z['validation']['ISIG']['all_E'],
          'imvo_train_pool_E':cost(sp,traincnt),
          'imvo_val_seen_E':cost(sp,val_seen['IMVO']),
          'isig_val_seen_E':cost(sp,val_seen['ISIG']),
          'initial_piece_count':len(has_space),'long_initial_piece_count':len(whole_words),
          'vocab_total_text_chars':alpha_count,'unique_training_types':z['unique_sampled_types']}
    arr.append(summary)

byband={}
for band in ['low','high']:
    group=[z for z in arr if z['band']==band]
    byband[band]={'n':len(group),'means':{k:float(statistics.mean(z[k] for z in group)) for k in group[0] if isinstance(group[0][k],(int,float)) and k!='seed'}}

pairs=collections.defaultdict(list)
for i in range(len(arr)):
    for j in range(i+1,len(arr)):
        a,b=arr[i],arr[j]
        typ='within_low' if a['band']==b['band']=='low' else 'within_high' if a['band']==b['band']=='high' else 'across'
        u,v=vocabs[a['name']],vocabs[b['name']]
        pairs[typ].append(len(u&v)/len(u|v))
ret={'band_stats':byband,'vocab_jaccard_by_pair_type':{k:{'mean':float(np.mean(v)),'sd':float(np.std(v,ddof=1)),'n':len(v)} for k,v in pairs.items()},
     'corr_trainpool_vs_unseen':float(np.corrcoef([z['imvo_train_pool_E'] for z in arr],[z['worst_unseen_E'] for z in arr])[0,1]),
     'models':arr,'method_scope':'EXPLORATORY descriptive analysis; mode threshold is post hoc'}
path=root/'mode_audit_results.json';path.write_text(json.dumps(ret,indent=2))
print('BANDS')
for k,v in byband.items():
 print(k,'n',v['n'], 'ETrain',round(v['means']['imvo_train_pool_E'],4),'EValNovel',round(v['means']['worst_unseen_E'],4),
       'EValSeen',round(v['means']['imvo_val_seen_E'],4),'InitPieces',round(v['means']['initial_piece_count'],1),
       'long initial',round(v['means']['long_initial_piece_count'],1),'trainingtypes',round(v['means']['unique_training_types'],1))
print('JACCARD',ret['vocab_jaccard_by_pair_type'])
print('CORR TRAINPOOL vs unseen',ret['corr_trainpool_vs_unseen'])
