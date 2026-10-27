#!/usr/bin/env python3
"""Evaluate pre-frozen new-sample selection on previously inspected newspaper validation."""
import collections,hashlib,json,math,numpy as np
from pathlib import Path
from classic_bpe_intervene import ClassicBPE
import run_phase9 as p
ROOT=p.ROOT;P8=p.p8
CP=ROOT/'fresh_selection_checkpoint.json'
SAVED_SHA=hashlib.sha256(CP.read_bytes()).hexdigest()
records=json.loads(CP.read_text())['runs']; assert len(records)==50
assert hashlib.sha256((ROOT/'FRESH_DRAW_RULE_PRECOMMIT.md').read_bytes()).hexdigest()==(ROOT/'FRESH_DRAW_RULE_PRECOMMIT.sha256').read_text().split()[0]
# Cross-family Phase9 shared denominator comes from original cached encoding arrays.
main=json.loads((ROOT/'results.json').read_text())['runs']
cache=[]
for r in main:
 with np.load(ROOT/'encodings'/(r['model_name']+'.npz')) as z: cache.append({k:z[k] for k in z.files})
seen={w for pg in P8.training_pages_cached for w in P8.forms(pg['text'])}
raw={d:collections.Counter(w for pg in P8.dev_pages() if pg['source']==d for w in P8.forms(pg['text'])) for d in ('IMVO','ISIG')}
lex={d:list(raw[d]) for d in raw}
weights={d:np.array([raw[d][w] for w in lex[d]],dtype=np.int64) for d in raw}
lens={d:np.array([P8.alpha_len(w) for w in lex[d]],dtype=np.int64) for d in raw}
novel={d:np.array([w not in seen for w in lex[d]],dtype=bool) for d in raw}
oldmask={d:np.logical_and.reduce([z[d+'_classic']!=-1 for z in cache]+[z[d+'_sp']!=-1 for z in cache]) for d in raw}
# Additional common support across all new fitted models, independent of their selection results.
allalph=[]
for r in records:
 for choice in r['choices']:
  j=json.loads((ROOT/'fresh_draws'/(choice['model_name']+'.json')).read_text())
  allalph.append(set(j['alphabet']))
shared=set.intersection(*allalph)
valid={d:oldmask[d]&np.array([all(c in shared for c in w) for w in lex[d]],dtype=bool) for d in raw}
support={d:{'valid_unseen_occurrences':int(sum(weights[d][valid[d]&novel[d]])),
 'all_unseen_occurrences':int(sum(weights[d][novel[d]])),
 'valid_unseen_alpha':int(sum(weights[d][valid[d]&novel[d]]*lens[d][valid[d]&novel[d]]))} for d in raw}


def score(model):
    rr={}
    for d in raw:
        mask=valid[d]&novel[d]
        ids=np.where(mask)[0]
        encoded=np.array([model.encode_len(lex[d][int(j)]) for j in ids],dtype=np.int32)
        assert min(encoded)>0
        rr[d]=100*int((encoded*weights[d][mask]).sum())/int((lens[d][mask]*weights[d][mask]).sum())
    return rr,max(rr.values())

# Never modify existing training-choice manifest; annotate separate file only.
scored=[]
for i,record in enumerate(records):
    choices=[]
    for choice in record['choices']:
        model=ClassicBPE.from_json(json.loads((ROOT/'fresh_draws'/(choice['model_name']+'.json')).read_text()))
        per,cost=score(model)
        choices.append({'option':choice['option'],'training_E':choice['training_E'],'worst_unseen_E':cost,
                        'per_domain_E':per,'first_merge':choice['pair']})
    natural=next(x for x in choices if x['option']=='greedy')
    selected=next(x for x in choices if x['option']==record['selected_option'])
    oracle=min(choices,key=lambda x:(x['worst_unseen_E'],x['option']))
    out={'seed':record['seed'],'mix':record['source_mix'],'trigger':record['trigger'],
         'selected_option':record['selected_option'],'selection_correct':selected['option']==oracle['option'],
         'natural_E':natural['worst_unseen_E'],'selected_E':selected['worst_unseen_E'],
         'oracle_E':oracle['worst_unseen_E'],'improvement':natural['worst_unseen_E']-selected['worst_unseen_E'],
         'choices':choices}
    scored.append(out)
    if (i+1)%10==0:print('FRESH_SCORED',i+1,'/50',flush=True)

rng=np.random.default_rng(20261023);B=10000
summary={}
for mix in [1.,.5]:
    subset=sorted([r for r in scored if r['mix']==mix],key=lambda x:x['seed'])
    nat=np.array([r['natural_E'] for r in subset]);sel=np.array([r['selected_E'] for r in subset])
    diff=nat-sel
    ids=rng.integers(0,len(subset),size=(B,len(subset)))
    ci=np.quantile(diff[ids].mean(axis=1),[.025,.975]).tolist()
    summary[f'mix_{int(mix*100)}']={'n':len(subset),'trigger_count':sum(x['trigger'] for x in subset),
        'train_selected_val_best_count':sum(x['selection_correct'] for x in subset),
        'natural_mean':float(nat.mean()),'natural_sd':float(nat.std(ddof=1)),
        'selected_mean':float(sel.mean()),'selected_sd':float(sel.std(ddof=1)),
        'natural_q90':float(np.quantile(nat,.9)),'selected_q90':float(np.quantile(sel,.9)),
        'mean_improvement':float(diff.mean()),'mean_improvement_bootstrap95':ci,
        'improved_count':int(sum(diff>1e-7)),'worsened_count':int(sum(diff< -1e-7)),
        'selected_first_merges':dict(collections.Counter(' + '.join(next(c['first_merge'] for c in r['choices'] if c['option']==r['selected_option'])) for r in subset))}
primary=summary['mix_100']
summary['precommitted_gate_pass']=(primary['train_selected_val_best_count']>=25 and primary['mean_improvement']>=.3 and primary['mean_improvement_bootstrap95'][0]>0 and primary['selected_sd']<=primary['natural_sd']*.25)
result={'protocol_sha256':hashlib.sha256((ROOT/'FRESH_DRAW_RULE_PRECOMMIT.md').read_bytes()).hexdigest(),
        'preselection_manifest_sha256':SAVED_SHA,
        'status':'NEW TRAINING DRAWS BUT RETROSPECTIVE REUSED VALIDATION; descriptive evidence only',
        'shared_support':support,'shared_alphabet':sorted(shared),'summary':summary,'records':scored}
(ROOT/'fresh_draw_validation_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print('SUMMARY',json.dumps(summary,ensure_ascii=False,indent=2))
