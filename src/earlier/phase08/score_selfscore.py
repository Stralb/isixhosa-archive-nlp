#!/usr/bin/env python3
"""Evaluate precommitted train-only selfcompression warning and best-of-three rule."""
import collections,hashlib,json,time
from pathlib import Path
import numpy as np
import sentencepiece as spm
import run_phase8 as r
st=time.monotonic()
proto_sha=hashlib.sha256((r.ROOT/'SELF_SCORE_PRECOMMIT.md').read_bytes()).hexdigest()
new=json.loads((r.ROOT/'selfscore_training_checkpoint.json').read_text())['runs'];assert len(new)==60
base=json.loads((r.ROOT/'training_checkpoint.json').read_text())['runs']
base=[x for x in base if x['method']=='word_uniform' and x['mix'] in (.5,1.0)]
assert len(base)==40
all_models=new+base
train=r.training_pages_cached
seen={w for p in train for w in r.forms(p['text'])}
original={d:collections.Counter(w for p in r.dev_pages() if p['source']==d for w in r.forms(p['text'])) for d in ('IMVO','ISIG')}
forms={d:list(original[d]) for d in original}
counts={d:np.array([original[d][w] for w in forms[d]],dtype=np.int64) for d in forms}
lens={d:np.array([r.alpha_len(w) for w in forms[d]],dtype=np.int64) for d in forms}
is_novel={d:np.array([w not in seen for w in forms[d]],dtype=bool) for d in forms}
alphabet={c for p in train for w in r.forms(p['text']) for c in w if c.isalpha()}
valid={d:np.array([all(not c.isalpha() or c in alphabet for c in w) for w in forms[d]],dtype=bool) for d in forms}
encoded=[]
for i,row in enumerate(all_models):
    m=spm.SentencePieceProcessor(model_file=str(r.MODELS/(row['name']+'.model')))
    out={}
    for d in forms:
        ee=m.encode(forms[d],out_type=int,num_threads=4)
        out[d]=np.array([len(v) if m.unk_id() not in v else -1 for v in ee],dtype=np.int32)
        valid[d]&=(out[d]!=-1)
    encoded.append(out)
    if (i+1)%25==0:print('BATCH',i+1,'/',len(all_models),round(time.monotonic()-st,2),flush=True)
support={}
for d in forms:
    support[d]={'total_words':int(sum(counts[d])),'supported_words':int(sum(counts[d][valid[d]])),
                'unseen_words':int(sum(counts[d][is_novel[d]])),
                'supported_unseen_words':int(sum(counts[d][is_novel[d]&valid[d]]))}
for i,row in enumerate(all_models):
    row['validation']={}
    for d in forms:
        row['validation'][d]={}
        for nm,mask in (('all',valid[d]),('unseen',valid[d]&is_novel[d])):
            numer=int(np.sum(counts[d][mask]*encoded[i][d][mask]));denom=int(np.sum(counts[d][mask]*lens[d][mask]))
            row['validation'][d][nm+'_E']=100*numer/denom
    row['validation']['worst_unseen_E']=max(row['validation'][d]['unseen_E'] for d in forms)
    row['validation']['worst_all_E']=max(row['validation'][d]['all_E'] for d in forms)

risk=np.array([x['validation']['worst_unseen_E']>=39.5 for x in new])
pred=np.array([x['self_train_E']>32.2 for x in new])
tp=int(sum(risk&pred));tn=int(sum(~risk&~pred));fp=int(sum(~risk&pred));fn=int(sum(risk&~pred))
assert (tp+tn+fp+fn)==60
sensitivity=tp/(tp+fn) if tp+fn else None
specificity=tn/(tn+fp) if tn+fp else None
bac=.5*(sensitivity+specificity) if sensitivity is not None and specificity is not None else None

triplets=[]
for j in range(20):
  three=new[j*3:j*3+3];assert [z['seed'] for z in three]==list(range(2001+3*j,2004+3*j))
  picked=min(three,key=lambda r:(r['self_train_E'],r['seed']))
  first=three[0]
  oracle=min(three,key=lambda r:r['validation']['worst_unseen_E'])
  triplets.append({'triplet':j+1,'first_seed':first['seed'],'selected_seed':picked['seed'],'oracle_seed':oracle['seed'],
     'baseline_E':first['validation']['worst_unseen_E'],
     'selected_E':picked['validation']['worst_unseen_E'],
     'oracle_E':oracle['validation']['worst_unseen_E'],
     'selected_self_train_E':picked['self_train_E'],
     'oracle_regret':picked['validation']['worst_unseen_E']-oracle['validation']['worst_unseen_E'],
     'selected_IMVO_E':picked['validation']['IMVO']['unseen_E'],
     'selected_ISIG_E':picked['validation']['ISIG']['unseen_E']})

baseline=np.array([x['baseline_E'] for x in triplets]); selected=np.array([x['selected_E'] for x in triplets]); delta=selected-baseline
rng=np.random.default_rng(20261025);B=10000;inds=rng.integers(0,20,size=(B,20))
q90=np.quantile(selected[inds],.9,axis=1)-np.quantile(baseline[inds],.9,axis=1)
meandiff=np.mean(delta[inds],axis=1)
q90_real=float(np.quantile(selected,.9)-np.quantile(baseline,.9))
mean_real=float(np.mean(delta))
ci_q90=np.quantile(q90,[.025,.975]).tolist();ci_mean=np.quantile(meandiff,[.025,.975]).tolist()

# Secondary fixed-source 50/50 group, already inspected, only retrospective comparator.
bal=np.array([x['validation']['worst_unseen_E'] for x in base if x['mix']==.5]);assert len(bal)==20
score_arr=np.array([z['self_train_E'] for z in new]); label_arr=np.array([z['validation']['worst_unseen_E'] for z in new]); type_arr=np.array([z['unique_sampled_types'] for z in new])
from scipy.stats import spearmanr
stats={'precommitted_threshold':32.2,'high_val_label_at_least':39.5,
 'confusion_matrix':{'true_positive':tp,'true_negative':tn,'false_positive':fp,'false_negative':fn},
 'balanced_accuracy':float(bac) if bac is not None else None,
 'sensitivity':sensitivity,'specificity':specificity,
 'train_only_correlation':float(np.corrcoef(score_arr,label_arr)[0,1]),
 'train_only_spearman':float(spearmanr(score_arr,label_arr).statistic),
 'unique_types_correlation':float(np.corrcoef(type_arr,label_arr)[0,1]),
 'triplet_baseline_mean':float(np.mean(baseline)),'triplet_selected_mean':float(np.mean(selected)),
 'triplet_baseline_q90':float(np.quantile(baseline,.9)),'triplet_selected_q90':float(np.quantile(selected,.9)),
 'pair_mean_diff_selected_minus_first':mean_real,'pair_mean_diff_bootstrap_95_ci':ci_mean,
 'q90_diff_selected_minus_first':q90_real,'q90_diff_bootstrap_95_ci':ci_q90,
 'selection_worse_than_baseline_count':int(sum(delta>0)),
 'selection_same_as_baseline_count':int(sum(delta==0)),
 'selected_equals_oracle_count':int(sum(a['selected_seed']==a['oracle_seed'] for a in triplets)),
 'mean_oracle_regret':float(np.mean([a['oracle_regret'] for a in triplets])),
 'practical_gate_pass':bool(q90_real<=-.40 and mean_real<=-.20 and ci_mean[1]<0),
 'classification_gate_pass':bool(bac is not None and bac>=.90 and ((fn/(fn+tp))<=.10 if (fn+tp) else False)),
 'reference_50_50_mean':float(np.mean(bal)),'reference_50_50_q90':float(np.quantile(bal,.9)),
 'observational_status':'New membership draws but NOT independent newspaper or uninspected validation split.'}
all_out={'precommit_sha256':proto_sha,'common_support':support,'metrics':stats,'triplets':triplets,
         'new_models':new,'baseline_models':base,'runtime_s':round(time.monotonic()-st,2)}
out=r.ROOT/'selfscore_results.json';out.write_text(json.dumps(all_out,indent=2))
print('CONFUSION',stats['confusion_matrix'],'balanced_acc',stats['balanced_accuracy'],'sens',stats['sensitivity'],'spec',stats['specificity'])
print('CORRELATIONS',stats['train_only_correlation'],stats['unique_types_correlation'])
print('TRIPLET baseline mean,q90',stats['triplet_baseline_mean'],stats['triplet_baseline_q90'])
print('TRIPLET selected mean,q90',stats['triplet_selected_mean'],stats['triplet_selected_q90'])
print('MEAN DIFF',mean_real,ci_mean,'Q90 DIFF',q90_real,ci_q90,'gate',stats['practical_gate_pass'])
print('50/50 old mean,q90',stats['reference_50_50_mean'],stats['reference_50_50_q90'])
print('DONE',out,round(time.monotonic()-st,2),flush=True)
