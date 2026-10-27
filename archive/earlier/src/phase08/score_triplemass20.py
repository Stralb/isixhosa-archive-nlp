#!/usr/bin/env python3
"""Common-support retrospective comparison to 20 high-mass single-fit baseline models."""
import collections,json,time
import numpy as np
import sentencepiece as spm
import run_phase8 as r
start=time.monotonic()
origin=json.load(open(r.ROOT/'selfscore_results.json'))
new=origin['new_models'];bal=[v for v in origin['baseline_models'] if v['mix']==.5]
assert len(new)==60 and len(bal)==20
triples=json.load(open(r.ROOT/'triplemass_training_checkpoint.json'))['runs'];assert len(triples)==20
full=json.load(open(r.ROOT/'large_single_training.json'))['runs'][1];assert full['tag']=='full_train_pool'
chosen=[min(new[3*i:3*i+3],key=lambda x:(x['self_train_E'],x['seed'])) for i in range(20)]
refs=[('best_of_3',x['name'],str(r.MODELS/(x['name']+'.model'))) for x in chosen]+[
('50_50',x['name'],str(r.MODELS/(x['name']+'.model'))) for x in bal]+[
('triple_mass_single',str(x['seed']),x['model_file']) for x in triples]+[
('full_train',full['tag'],full['model_file'])]
seen={w for p in r.training_pages_cached for w in r.forms(p['text'])}
raw={d:collections.Counter(w for p in r.dev_pages() if p['source']==d for w in r.forms(p['text'])) for d in ('IMVO','ISIG')}
lex={d:list(raw[d]) for d in raw}
cnt={d:np.array([raw[d][w] for w in lex[d]],dtype=int) for d in raw}
num={d:np.array([r.alpha_len(w) for w in lex[d]],dtype=int) for d in raw}
unseen={d:np.array([w not in seen for w in lex[d]],dtype=bool) for d in raw}
alphabet={c for p in r.training_pages_cached for w in r.forms(p['text']) for c in w if c.isalpha()}
valid={d:np.array([all(not c.isalpha() or c in alphabet for c in w) for w in lex[d]],dtype=bool) for d in raw}
evals=[]
for i,(_,name,path) in enumerate(refs):
  p=spm.SentencePieceProcessor(model_file=path);stat={}
  for d in raw:
    v=p.encode(lex[d],out_type=int,num_threads=4)
    stat[d]=np.array([len(ids) if p.unk_id() not in ids else -1 for ids in v],dtype=int)
    valid[d]&=(stat[d]>=0)
  evals.append(stat)
score=[]
for z in evals:
    zz={}
    for d in raw:
        zz[d]={}
        for k,mask in [('all',valid[d]),('unseen',valid[d]&unseen[d])]:
            zz[d][k+'_E']=float(100*np.sum(cnt[d][mask]*z[d][mask])/np.sum(cnt[d][mask]*num[d][mask]))
    zz['worst_unseen_E']=max(zz[d]['unseen_E'] for d in raw)
    score.append(zz)

def make(name):
    v=np.array([score[i]['worst_unseen_E'] for i,rr in enumerate(refs) if rr[0]==name])
    return {'n':len(v),'values':v.tolist(),'mean':float(np.mean(v)),'std':float(np.std(v,ddof=1)) if len(v)>1 else None,
            'q90':float(np.quantile(v,.9)),'range':[float(np.min(v)),float(np.max(v))],
            'high_risk_count_over_39_5':int(sum(v>=39.5))}
summary={q:make(q) for q in ('best_of_3','50_50','triple_mass_single','full_train')}
rng=np.random.default_rng(20261026);B=10000
A=np.array(summary['best_of_3']['values']);C=np.array(summary['triple_mass_single']['values'])
ia=rng.integers(0,20,size=(B,20));ic=rng.integers(0,20,size=(B,20))
mean_d=np.mean(A[ia],axis=1)-np.mean(C[ic],axis=1)
q90_d=np.quantile(A[ia],.9,axis=1)-np.quantile(C[ic],.9,axis=1)
summary['best3_minus_triplemass']={
    'mean_diff':float(np.mean(A)-np.mean(C)),
    'mean_independent_bootstrap95':np.quantile(mean_d,[.025,.975]).tolist(),
    'q90_diff':float(np.quantile(A,.9)-np.quantile(C,.9)),
    'q90_independent_bootstrap95':np.quantile(q90_d,[.025,.975]).tolist()}
summary['support']={d:{'total_words':int(np.sum(cnt[d])),'supported_words':int(np.sum(cnt[d][valid[d]])),
       'unseen_words':int(np.sum(cnt[d][unseen[d]])),
       'supported_unseen_words':int(np.sum(cnt[d][unseen[d]&valid[d]]))} for d in raw}
summary['individual_full_costs']=score[-1]
summary['runtime_s']=time.monotonic()-start
summary['note']='All results retrospective; 20 x 1.265M sampled single fit vs 20 best-of-three 421696-char fits and one 1.406M full pool.'
p=r.ROOT/'triplemass_results.json';p.write_text(json.dumps(summary,indent=2))
for k,x in summary.items():
 if isinstance(x,dict) and 'values' in x:print(k,'n',x['n'],'mean',round(x['mean'],4),'q90',round(x['q90'],4),'sd',x['std'],'hi_risk',x['high_risk_count_over_39_5'])
print('comparison',summary['best3_minus_triplemass'])
print('support',summary['support'])
print('DONE',p)
