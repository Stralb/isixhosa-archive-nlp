#!/usr/bin/env python3
"""Phase 10: precommitted token-level neural LM comparison on fixed newspaper text.

Usage:
  python neural_merge_test.py prepare
  python neural_merge_test.py fit 0 3
  python neural_merge_test.py analyze

No validation or external results are used to modify model training or fit selection.
"""
import argparse, hashlib, json, math, os, random, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
P9=Path('/mnt/data/isiXhosa_sampling_mechanism_phase9')
P8=Path('/mnt/data/isiXhosa_sampling_stability_phase8')
COMMON=Path('/mnt/data/isiXhosa_historical_eval/isiXhosa_external_validation')
assert hashlib.sha256((ROOT/'PHASE10_PRECOMMIT.md').read_bytes()).hexdigest()==(ROOT/'PHASE10_PRECOMMIT.sha256').read_text().split()[0]
sys.path.insert(0,str(P9));sys.path.insert(0,str(COMMON))
from classic_bpe_intervene import ClassicBPE
from common import training_pages,dev_pages,forms,alpha_len

SEEDS=(101,202,303)
CONTEXT=6; DIM=48; HIDDEN=128; STEPS=1200; BATCH=256
PREP=ROOT/'cache'; PREP.mkdir(exist_ok=True)
MODEL_DIR=ROOT/'neural_models';MODEL_DIR.mkdir(exist_ok=True)
SPECS=[
 ('imvo5001_greedy',P9/'fresh_draws/seed_5001_mix_100_greedy.json'),
 ('imvo5001_second',P9/'fresh_draws/seed_5001_mix_100_second.json'),
 ('imvo5004_greedy',P9/'fresh_draws/seed_5004_mix_100_greedy.json'),
 ('imvo5004_second',P9/'fresh_draws/seed_5004_mix_100_second.json'),
 ('imvo5012_greedy',P9/'fresh_draws/seed_5012_mix_100_greedy.json'),
 ('imvo5012_second',P9/'fresh_draws/seed_5012_mix_100_second.json'),
 ('balanced6001',P9/'fresh_draws/seed_6001_mix_50_greedy.json'),
 ('fullpool_imvo',ROOT/'fullpool_imvo_classic.json'),
]
TASKS=[(name,seed) for name,_ in SPECS for seed in SEEDS]


def write_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2))


def get_model(name):
    fn=dict(SPECS)[name]
    return ClassicBPE.from_json(json.loads(fn.read_text(encoding='utf8')))


def all_documents():
    # Original period used for LM training, independent of tokenizer training sample draw.
    corpus_pages=training_pages()
    sampled=[];selection={}
    for source in ['IMVO','ISIG']:
        candidates=sorted([p for p in corpus_pages if p['source']==source],
            key=lambda p:hashlib.sha256(('phase10-lm-v1|'+p['id']).encode()).hexdigest())
        chars=0
        chosen=[]
        for page in candidates:
            chosen.append(page)
            chars+=sum(alpha_len(w) for w in forms(page['text']))
            if chars>=300000:break
        assert chars>=300000
        sampled.extend(chosen)
        selection[source]={'page_ids':sorted(p['id'] for p in chosen),'pages':len(chosen),'alpha_chars_raw':chars}
    # TRAIN pages ordered source then lexicographic date; preserves within-page word order.
    sampled.sort(key=lambda p:(p['source'],p['id']))
    train=[{'id':p['id'],'source':p['source'],'text':p['text']} for p in sampled]
    evaluation={}
    for source in ('IMVO','ISIG'):
        evaluation[source]=[{'id':p['id'],'text':p['text']} for p in dev_pages() if p['source']==source]
    evaluation['ityala']=[{'id':'ityala_ocr_full','text':Path('/mnt/data/isiXhosa_historical_eval/ocr/ityala_lamawele_raw.txt').read_text()}]
    evaluation['zem']=[{'id':'zem_ocr_full','text':Path('/mnt/data/isiXhosa_historical_eval/ocr/zemkinkomo_magwalandini_raw.txt').read_text()}]
    items=[json.loads(s) for s in Path('/mnt/data/xho.jsonl').read_text(encoding='utf8').splitlines() if s.strip()]
    evaluation['modern']=[{'id':f'modern_{i:04d}','text':j['text']} for i,j in enumerate(items)
                   if isinstance(j.get('text'),str) and alpha_len(j['text'])>0]
    assert len(evaluation['modern'])==181
    return train,evaluation,selection


def fullpool_fit():
    src=[p for p in training_pages() if p['source']=='IMVO']
    ww=[w for p in src for w in forms(p['text'])]
    assert len(ww)>200000
    t=time.monotonic()
    bp=ClassicBPE.fit(ww,vocab_size=2000)
    payload=bp.to_json();payload['training_source']='all 1887 Jan-Sep IMVO word occurrences'
    payload['training_alpha']=sum(alpha_len(w) for w in ww)
    payload['training_words']=len(ww)
    write_json(ROOT/'fullpool_imvo_classic.json',payload)
    print('FULLPOOL_FIT',len(ww),'words,',payload['training_alpha'],'chars',round(time.monotonic()-t,1),'sec',flush=True)


def make_vocabulary(bp):
    voc=sorted(set(bp.alphabet)|{''.join(p) for p in bp.merges})
    return {v:i+2 for i,v in enumerate(voc)} # PAD=0 BOS=1, no UNK in scored text


def make_data(docs,bp,common_alpha,vocab):
    X=[];Y=[];info=[]
    support_n=0;total_n=0;support_chars=0;all_chars=0
    for doc in docs:
        start=len(Y);ctx=[1]*CONTEXT
        den=0;words=0;allwords=0
        for w in forms(doc['text']):
            allwords+=1;total_n+=1;all_chars+=alpha_len(w)
            if any(c not in common_alpha for c in w):
                ctx=[1]*CONTEXT
                continue
            pieces=bp.pieces(w)
            if pieces is None:
                raise AssertionError(f'Common supported word failed model encoding {w}')
            words+=1;support_n+=1;n=alpha_len(w);support_chars+=n;den+=n
            for x in pieces:
                tok=vocab[x]
                X.append(ctx.copy());Y.append(tok)
                ctx=ctx[1:]+[tok]
        info.append({'id':doc['id'],'start':start,'stop':len(Y),'alpha_chars':den,
             'words':words,'all_words':allwords})
    return (np.array(X,dtype=np.int16).reshape(-1,CONTEXT),np.array(Y,dtype=np.int16),info,
          {'word_supported':support_n,'words_total':total_n,'alpha_supported':support_chars,'alpha_total':all_chars,
           'tokens':len(Y),'documents':len(docs)})


def prepare():
    if not (ROOT/'fullpool_imvo_classic.json').exists():fullpool_fit()
    train,ev,selection=all_documents()
    bpes={name:get_model(name) for name,_ in SPECS}
    common_alpha=set.intersection(*[set(b.alphabet) for b in bpes.values()])
    assert len(common_alpha)>26
    source_sha=hashlib.sha256(Path('/mnt/data/TxtLangFiles1870-1894_TB.zip').read_bytes()).hexdigest()
    checks={'precommit_sha256':hashlib.sha256((ROOT/'PHASE10_PRECOMMIT.md').read_bytes()).hexdigest(),
       'newspaper_zip_sha256':source_sha,'training_selection':selection,'shared_char_alphabet':sorted(common_alpha),
       'model_paths':{k:str(v) for k,v in SPECS},'models':{},'eval_corpora':list(ev)}
    for name,b in bpes.items():
        t=time.monotonic();voc=make_vocabulary(b);checks['models'][name]={'vocab':len(voc)+2,'first_merge':b.merges[0]}
        for g,docs in [('TRAIN',train),*ev.items()]:
            meta=PREP/f'{name}_{g}.json';datafile=PREP/f'{name}_{g}.npz'
            if meta.exists() and datafile.exists():
                record=json.loads(meta.read_text());stats=record['stats']
            else:
                X,y,di,stats=make_data(docs,b,common_alpha,voc)
                np.savez_compressed(datafile,X=X,y=y)
                write_json(meta,{'info':di,'stats':stats,'num_targets':len(y)})
            if g!='TRAIN':
                assert stats['alpha_supported']>0
                # Same normalization, document list, and common support for every compared tokenizer
                if name==SPECS[0][0]:checks.setdefault('corpora',{})[g]={'stats':stats,'doc_hash':hashlib.sha256('\n'.join(d['id'] for d in docs).encode()).hexdigest()}
                else:
                    ref=checks['corpora'][g]['stats']
                    assert all(stats[k]==ref[k] for k in ['word_supported','words_total','alpha_supported','alpha_total','documents'])
        checks['models'][name]['train_tokens']=json.loads((PREP/f'{name}_TRAIN.json').read_text())['num_targets']
        print('PREPARED',name,'vocab',len(voc)+2,'train_tokens',checks['models'][name]['train_tokens'],
              'seconds',round(time.monotonic()-t,1),flush=True)
    write_json(ROOT/'PREPARED_MANIFEST.json',checks)
    print('PREPARE_DONE',len(SPECS),'tokenizers',len(common_alpha),'shared symbols',flush=True)


class FFTokenLM(nn.Module):
    def __init__(self,v):
        super().__init__()
        self.embedding=nn.Embedding(v,DIM)
        self.fc1=nn.Linear(CONTEXT*DIM,HIDDEN)
        self.drop=nn.Dropout(.10)
        self.fc2=nn.Linear(HIDDEN,v)
    def forward(self,x):
        z=self.embedding(x).reshape(x.size(0),-1)
        z=self.drop(F.gelu(self.fc1(z)))
        return self.fc2(z)


def prepare_tensors(name,corp):
    with np.load(PREP/f'{name}_{corp}.npz') as z:
        X=torch.from_numpy(z['X'].astype(np.int64));y=torch.from_numpy(z['y'].astype(np.int64))
    return X,y


@torch.no_grad()
def evaluate(model,X,Y,metadata):
    model.eval()
    L=np.empty(len(Y),dtype=np.float32)
    for start in range(0,len(Y),4096):
        stop=min(len(Y),start+4096)
        logits=model(X[start:stop]);e=F.cross_entropy(logits,Y[start:stop],reduction='none')
        L[start:stop]=e.detach().cpu().numpy()
    table=json.loads(metadata.read_text())['info']
    total_nat=float(L.sum(dtype=np.float64));total_chars=sum(z['alpha_chars'] for z in table)
    bpc=total_nat/math.log(2)/total_chars
    per_doc=[]
    for z in table:
        chars=z['alpha_chars'];start=z['start'];stop=z['stop']
        if chars:
            sub=float(L[start:stop].sum(dtype=np.float64))
            per_doc.append({'id':z['id'],'nats':sub,'chars':chars,'bpc':sub/math.log(2)/chars,'tokens':stop-start})
    return {'bpc':bpc,'nats':total_nat,'alpha_chars':total_chars,'tokens':len(Y),'doc':per_doc}


def fit_range(start,stop):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    assert (ROOT/'PREPARED_MANIFEST.json').exists()
    assert 0<=start<stop<=len(TASKS)
    for ix in range(start,stop):
        name,seed=TASKS[ix];fn=ROOT/f'run_{name}_{seed}.json'
        if fn.exists():
            print('REPLAY_ALREADY',ix,name,seed,flush=True);continue
        t=time.monotonic();v=json.loads((ROOT/'PREPARED_MANIFEST.json').read_text())['models'][name]['vocab']
        torch.manual_seed(seed);rng=np.random.default_rng(seed+90210)
        model=FFTokenLM(v);optimizer=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.01)
        X,Y=prepare_tensors(name,'TRAIN');N=len(Y);logs=[]
        perm=rng.permutation(N);offset=0
        model.train()
        for step in range(1,STEPS+1):
            if offset+BATCH>N:
                perm=rng.permutation(N);offset=0
            sel=perm[offset:offset+BATCH];offset+=BATCH
            xx=X[sel]; yy=Y[sel]
            optimizer.zero_grad(set_to_none=True)
            out=model(xx);loss=F.cross_entropy(out,yy)
            loss.backward();optimizer.step()
            if step in (1,300,600,900,1200):logs.append({'step':step,'training_ce_nats':float(loss.detach())})
        model.eval()
        outcomes={}
        for group in ('IMVO','ISIG','ityala','zem','modern'):
            ex,ey=prepare_tensors(name,group)
            outcomes[group]=evaluate(model,ex,ey,PREP/f'{name}_{group}.json')
        outcomes['newspaper_macro_bpc']=(outcomes['IMVO']['bpc']+outcomes['ISIG']['bpc'])/2
        outcomes['newspaper_worst_bpc']=max(outcomes['IMVO']['bpc'],outcomes['ISIG']['bpc'])
        # Save enough to replay evaluation; no model-based validation/early stopping.
        model_path=MODEL_DIR/f'{name}_seed{seed}.pt'
        torch.save(model.state_dict(),model_path)
        res={'task_index':ix,'tokenizer':name,'neural_seed':seed,
            'trained_steps':STEPS,'batch_size':BATCH,'train_targets_available':N,
            'tokens_seen_with_replacement_across_epochs':STEPS*BATCH,
            'architecture':{'context':CONTEXT,'embed':DIM,'hidden':HIDDEN,'vocab':v,'dropout':.10},
            'learning_rate':.0015,'train_loss_log':logs,'evaluation':outcomes,
            'fit_seconds':round(time.monotonic()-t,2),
            'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest()}
        write_json(fn,res)
        print('TRAINED',ix+1,'/',len(TASKS),name,seed,'seconds',res['fit_seconds'],
             'val_IMVO',round(outcomes['IMVO']['bpc'],4),'val_ISIG',round(outcomes['ISIG']['bpc'],4),
             'macro',round(outcomes['newspaper_macro_bpc'],4),'worst',round(outcomes['newspaper_worst_bpc'],4),flush=True)


def analyze():
    import statistics
    rows=[]
    for name,seed in TASKS:
        fn=ROOT/f'run_{name}_{seed}.json'
        assert fn.exists(),fn
        rows.append(json.loads(fn.read_text()))
    by={(x['tokenizer'],x['neural_seed']):x for x in rows}
    groups=['IMVO','ISIG','newspaper_macro_bpc','newspaper_worst_bpc','ityala','zem','modern']
    def get(x,group):
        return x['evaluation'][group] if group.startswith('newspaper_') else x['evaluation'][group]['bpc']
    summar={}
    for name,_ in SPECS:
        summar[name]={'fit_seed_values':{g:[get(by[name,s],g) for s in SEEDS] for g in groups}}
        summar[name]['mean_bpc']={g:float(np.mean(summar[name]['fit_seed_values'][g])) for g in groups}
    contrasts=[]
    comparisons=[
       ('5004_selected_second_vs_greedy','imvo5004_second','imvo5004_greedy'),
       ('5001_good_greedy_vs_bad_second','imvo5001_greedy','imvo5001_second'),
       ('5012_good_greedy_vs_bad_second','imvo5012_greedy','imvo5012_second'),
       ('5004_selected_vs_balanced','imvo5004_second','balanced6001'),
       ('5004_selected_vs_fullpool','imvo5004_second','fullpool_imvo'),
       ('balanced_vs_fullpool','balanced6001','fullpool_imvo'),
    ]
    rng=np.random.default_rng(20261023)
    for label,a,b in comparisons:
        contrasts_obj={'name':label,'a':a,'b':b,'metrics':{}}
        for group in groups:
            differences=np.array([get(by[a,s],group)-get(by[b,s],group) for s in SEEDS])
            draw=rng.integers(0,len(SEEDS),(10000,len(SEEDS)))
            ci=np.quantile(differences[draw].mean(axis=1),[.025,.975]).tolist()
            contrasts_obj['metrics'][group]={'a_minus_b_mean':float(np.mean(differences)),
                 'paired_per_seed':differences.tolist(),
                 'descriptive_bootstrap_seed95':ci,
                 'a_better_neural_seeds':int(np.sum(differences<0))}
        contrasts.append(contrasts_obj)
    p5004=contrasts[0]['metrics'];p5001=contrasts[1]['metrics']
    def works(metrics):
        return metrics['a_minus_b_mean']<=-.02 and metrics['a_better_neural_seeds']>=2
    gate=works(p5004['newspaper_worst_bpc']) and works(p5001['newspaper_worst_bpc'])
    out={'status':'retrospective corpus-level neural downstream check; new neural model fits',
      'precommit_sha256':hashlib.sha256((ROOT/'PHASE10_PRECOMMIT.md').read_bytes()).hexdigest(),
      'setup':{'lm_neural_seeds':SEEDS,'fixed_steps':STEPS,'batch':BATCH,'context':CONTEXT,'dropout':.10},
      'models':summar,'contrasts':contrasts,'primary_gate_pass':bool(gate)}
    write_json(ROOT/'PHASE10_RESULTS.json',out)
    print('PRIMARY_GATE_PASS',gate)
    for c in contrasts:
        print(c['name'],'Worst bpc difference',round(c['metrics']['newspaper_worst_bpc']['a_minus_b_mean'],5),
              'Ityala',round(c['metrics']['ityala']['a_minus_b_mean'],5),
              'Zem',round(c['metrics']['zem']['a_minus_b_mean'],5),
              'Modern',round(c['metrics']['modern']['a_minus_b_mean'],5),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','fit','analyze']);parser.add_argument('start',type=int,nargs='?',default=0);parser.add_argument('stop',type=int,nargs='?',default=3)
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='fit':fit_range(args.start,args.stop)
    else:analyze()
