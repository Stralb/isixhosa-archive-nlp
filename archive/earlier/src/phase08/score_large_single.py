#!/usr/bin/env python3
"""Compute-comparable single-fit BPE baselines vs multi-start at common support."""
import collections,json,time
import numpy as np
import sentencepiece as spm
import run_phase8 as r
st=time.monotonic()
a=json.load(open(r.ROOT/'selfscore_results.json'))
rows=a['new_models']+a['baseline_models']
special=json.load(open(r.ROOT/'large_single_training.json'))['runs']
assert len(rows)==100
seen={w for p in r.training_pages_cached for w in r.forms(p['text'])}
vals={s:collections.Counter(w for p in r.dev_pages() if p['source']==s for w in r.forms(p['text'])) for s in ('IMVO','ISIG')}
keys={s:list(vals[s]) for s in vals}
n={s:np.array([vals[s][w] for w in keys[s]],dtype=int) for s in vals}
alpha={s:np.array([r.alpha_len(w) for w in keys[s]],dtype=int) for s in vals}
novel={s:np.array([w not in seen for w in keys[s]],dtype=bool) for s in vals}
trainalphabet={c for p in r.training_pages_cached for w in r.forms(p['text']) for c in w if c.isalpha()}
valid={s:np.array([all(not c.isalpha() or c in trainalphabet for c in w) for w in keys[s]]) for s in keys}
proc_paths=[str(r.MODELS/(z['name']+'.model')) for z in rows]+[z['model_file'] for z in special]
enc=[]
for path in proc_paths:
   p=spm.SentencePieceProcessor(model_file=path)
   ee={}
   for s in keys:
      z=p.encode(keys[s],out_type=int,num_threads=4)
      ee[s]=np.array([len(ids) if p.unk_id() not in ids else -1 for ids in z],dtype=int)
      valid[s]&=(ee[s]>=0)
   enc.append(ee)
score=[]
for ee in enc:
  d={}
  for s in keys:
      d[s]={}
      for kind,mask in [('unseen',valid[s]&novel[s]),('all',valid[s])]:
        numer=np.sum(n[s][mask]*ee[s][mask]);denom=np.sum(n[s][mask]*alpha[s][mask]);d[s][kind+'_E']=float(100*numer/denom)
  d['worst_unseen_E']=max(d[s]['unseen_E'] for s in keys)
  score.append(d)
B=60
new=rows[:B]
chosen=[min(new[3*i:3*i+3],key=lambda rr:(rr['self_train_E'],rr['seed'])) for i in range(20)]
name_to_pos={z['name']:i for i,z in enumerate(rows)}
choice_vals=np.array([score[name_to_pos[z['name']]]['worst_unseen_E'] for z in chosen])
bal_vals=np.array([score[i]['worst_unseen_E'] for i,z in enumerate(rows) if z.get('mix')==.5])
large_res={z['tag']:score[len(rows)+i] for i,z in enumerate(special)}
final={'method':'EXPLORATORY strong budget-matched single BPE baseline',
       'support':{s:{'total':int(np.sum(n[s])),'eligible':int(np.sum(n[s][valid[s]])),
                     'unseen':int(np.sum(n[s][novel[s]])),
                     'eligible_unseen':int(np.sum(n[s][valid[s]&novel[s]]))} for s in keys},
       'single_models':large_res,
       'best_of_three':{'mean':float(np.mean(choice_vals)),'q90':float(np.quantile(choice_vals,.9)),
                        'model_character_mass_per_three_training_runs':3*r.BUDGET},
       'reference_50_50':{'mean':float(np.mean(bal_vals)),'q90':float(np.quantile(bal_vals,.9))},
       'newly_fitted':special,
       'note':'Post-hoc single BPE comparison; one-shot scores cannot measure its sampling variance.'}
(r.ROOT/'large_single_results.json').write_text(json.dumps(final,indent=2))
print('BESTOF3',final['best_of_three'])
print('50/50',final['reference_50_50'])
print('SINGLES')
for key,item in large_res.items():print(key,item)
print('SUPPORT',final['support'])
print('runtime',time.monotonic()-st)
