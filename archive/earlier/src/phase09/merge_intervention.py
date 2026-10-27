#!/usr/bin/env python3
import collections,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from classic_bpe_intervene import ClassicBPE
import run_phase9 as p
ROOT=p.ROOT
assert hashlib.sha256((ROOT/'MERGE_INTERVENTION_PROTOCOL.md').read_bytes()).hexdigest()==(ROOT/'MERGE_INTERVENTION_PROTOCOL.sha256').read_text().split()[0]
P8=p.p8
allprior=json.loads((ROOT/'results.json').read_text())
# Recreate precisely the intersection mask used by phase9 scorer.
rows=allprior['runs'];CACHED=ROOT/'encodings'
train={w for pg in p.p8.training_pages_cached for w in p.p8.forms(pg['text'])}
raw={d:collections.Counter(w for pg in p.p8.dev_pages() if pg['source']==d for w in p.p8.forms(pg['text'])) for d in ('IMVO','ISIG')}
lex={d:list(raw[d]) for d in raw}
weights={d:np.array([raw[d][w] for w in lex[d]],dtype=np.int64) for d in raw}
lens={d:np.array([P8.alpha_len(w) for w in lex[d]],dtype=np.int64) for d in raw}
novel={d:np.array([w not in train for w in lex[d]],dtype=bool) for d in raw}
cache=[]
for r in rows:
    with np.load(CACHED/(r['model_name']+'.npz')) as z:cache.append({k:z[k] for k in z.files})
valid={d:np.logical_and.reduce([z[d+'_classic']!=-1 for z in cache]+[z[d+'_sp']!=-1 for z in cache]) for d in raw}
for d in raw:
    n=int((weights[d][valid[d]&novel[d]]*lens[d][valid[d]&novel[d]]).sum())
    assert n==allprior['summary']['support'][d]['valid_unseen_alpha']

baseline={}
for seed in (1001,1006):
    basefile=ROOT/'models'/f'classic_mix100_draw{seed}.json'
    baseline[seed]=ClassicBPE.from_json(json.loads(basefile.read_text()))

k_values=[0,1,5,20,100,300,'self100']
results=[]

def worst_cost(model):
    data={}
    for d in raw:
        mask=valid[d]&novel[d]
        w=weights[d][mask];ln=lens[d][mask]
        counts=np.array([model.encode_len(x) for x,keep in zip(lex[d],mask) if keep],dtype=np.int32)
        assert np.all(counts>0),d
        data[d]=100*int((counts*w).sum())/int((ln*w).sum())
    return data,max(data.values())

def voc(model):return model.alphabet|{''.join(merge) for merge in model.merges}

def main():
    for target,donor in ((1001,1006),(1006,1001)):
        words=p.get_sample(1.,target)
        donor_model=baseline[donor]
        base_model=baseline[target]
        basev=voc(base_model);donv=voc(donor_model)
        base_score=max(r['validation']['worst_unseen_E'] for r in rows if r['mix']==1. and r['seed']==target)
        donor_score=max(r['validation']['worst_unseen_E'] for r in rows if r['mix']==1. and r['seed']==donor)
        for k in k_values:
            if k=='self100':forced=base_model.merges[:100]
            else:forced=donor_model.merges[:k]
            start=time.monotonic()
            out={'target_seed':target,'donor_seed':donor,'forced_prefix_count':k,
                 'target_baseline_E':base_score,'donor_baseline_E':donor_score}
            try:
                model=ClassicBPE.fit(words,vocab_size=2000,forced_prefix=forced)
                if k==0 or k=='self100':
                    assert model.merges==base_model.merges,('baseline self control failed',target,k)
                val,worst=worst_cost(model)
                vocab=voc(model)
                out.update({'valid':True,'cost_by_domain':val,'worst_unseen_E':worst,
                    'change_from_own_baseline':worst-base_score,
                    'towards_donor_cost':abs(base_score-donor_score)-abs(worst-donor_score),
                    'vocab_jaccard_with_donor':len(vocab&donv)/len(vocab|donv),
                    'baseline_jaccard_with_donor':len(basev&donv)/len(basev|donv),
                    'jaccard_gain_to_donor':len(vocab&donv)/len(vocab|donv)-len(basev&donv)/len(basev|donv),
                    'merges':len(model.merges),'walltime_s':round(time.monotonic()-start,2)})
                dest=ROOT/'intervention_models';dest.mkdir(exist_ok=True)
                name=f'target_{target}_donor_{donor}_prefix_{k}.json'
                (dest/name).write_text(json.dumps(model.to_json(),ensure_ascii=False))
            except ValueError as e:
                out.update({'valid':False,'reason':str(e),'walltime_s':round(time.monotonic()-start,2)})
            results.append(out)
            (ROOT/'merge_intervention_results.json').write_text(json.dumps({'protocol_sha256':hashlib.sha256((ROOT/'MERGE_INTERVENTION_PROTOCOL.md').read_bytes()).hexdigest(),
             'groups':'1001 low cost; 1006 high cost on prior already viewed validation',
             'records':results},indent=2))
            if out['valid']:
                print('MERGE_COUNTERFACTUAL',target,'->',donor,'k',k,'E',round(out['worst_unseen_E'],4),
                       'frombase',round(out['change_from_own_baseline'],4),'jac_gain',round(out['jaccard_gain_to_donor'],4),flush=True)
            else:print('INVALID',target,donor,k,out['reason'],flush=True)
    ok=[]
    for target in (1001,1006):
        cases=[r for r in results if r['target_seed']==target and r['valid'] and isinstance(r['forced_prefix_count'],int) and 1<=r['forced_prefix_count']<=100]
        ok.append(any(r['towards_donor_cost']>=.7 and r['jaccard_gain_to_donor']>=.1 for r in cases))
    print('MECHANISTIC_GATE_BOTH_RECIPROCAL',all(ok),ok,flush=True)

if __name__=='__main__':main()
