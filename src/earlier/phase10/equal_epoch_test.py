#!/usr/bin/env python3
"""Secondary posthoc robustness: exact two passes over each tokenized fixed raw corpus."""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import neural_merge_test as phase
ROOT=phase.ROOT
assert hashlib.sha256((ROOT/'EQUAL_EPOCH_PROTOCOL.md').read_bytes()).hexdigest()==(ROOT/'EQUAL_EPOCH_PROTOCOL.sha256').read_text().split()[0]
MODELS=('imvo5001_greedy','imvo5001_second','imvo5004_greedy','imvo5004_second')
TASKS=[(m,s) for m in MODELS for s in phase.SEEDS]

def fit(start,stop):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    for ix in range(start,stop):
        name,seed=TASKS[ix];out=ROOT/f'epoch_{name}_{seed}.json'
        if out.exists():print('EXISTS',ix,name,seed,flush=True);continue
        t=time.monotonic();v=json.loads((ROOT/'PREPARED_MANIFEST.json').read_text())['models'][name]['vocab']
        torch.manual_seed(seed);rng=np.random.default_rng(seed+190221)
        model=phase.FFTokenLM(v);optimizer=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.01)
        X,Y=phase.prepare_tensors(name,'TRAIN');N=len(Y);updates=0;logs=[]
        model.train()
        for epoch in range(2):
            perm=rng.permutation(N)
            losses=[]
            for offset in range(0,N,phase.BATCH):
                indices=perm[offset:offset+phase.BATCH]
                optimizer.zero_grad(set_to_none=True)
                loss=F.cross_entropy(model(X[indices]),Y[indices])
                loss.backward();optimizer.step();updates+=1
                losses.append(float(loss.detach()))
            logs.append({'epoch':epoch+1,'steps':updates,'mean_batch_training_ce_nats':float(np.mean(losses))})
        model.eval();outcomes={}
        for corpus in ('IMVO','ISIG','ityala','zem','modern'):
            ex,ey=phase.prepare_tensors(name,corpus)
            outcomes[corpus]=phase.evaluate(model,ex,ey,phase.PREP/f'{name}_{corpus}.json')
        outcomes['newspaper_macro_bpc']=(outcomes['IMVO']['bpc']+outcomes['ISIG']['bpc'])/2
        outcomes['newspaper_worst_bpc']=max(outcomes['IMVO']['bpc'],outcomes['ISIG']['bpc'])
        model_file=phase.MODEL_DIR/f'epoch_{name}_{seed}.pt';torch.save(model.state_dict(),model_file)
        phase.write_json(out,{'task_index':ix,'model':name,'neural_seed':seed,'epochs':2,
            'optimizer_steps':updates,'training_targets_per_epoch':N,'loss_log':logs,'evaluation':outcomes,
            'model_sha256':hashlib.sha256(model_file.read_bytes()).hexdigest(),
            'seconds':round(time.monotonic()-t,2)})
        print('EPOCH_TRAINED',ix+1,'/',len(TASKS),name,seed,'steps',updates,
             'worst',round(outcomes['newspaper_worst_bpc'],5),'macro',round(outcomes['newspaper_macro_bpc'],5),
             'seconds',round(time.monotonic()-t,2),flush=True)

def analyze():
    seeds=phase.SEEDS
    groups=['IMVO','ISIG','newspaper_macro_bpc','newspaper_worst_bpc','ityala','zem','modern']
    allruns={(m,s):json.loads((ROOT/f'epoch_{m}_{s}.json').read_text()) for m,s in TASKS}
    def loss(run,g):return run['evaluation'][g] if g.startswith('newspaper_') else run['evaluation'][g]['bpc']
    report={}
    for root,a,b in [('5001','imvo5001_greedy','imvo5001_second'),('5004','imvo5004_second','imvo5004_greedy')]:
        record={}
        for grp in groups:
            ds=[loss(allruns[a,s],grp)-loss(allruns[b,s],grp) for s in seeds]
            record[grp]={'ng_minus_startn_per_seed':ds,'mean':float(np.mean(ds)),
                'ng_better_count':sum(d<0 for d in ds)}
        report[root]=record
    passed=all(report[k]['newspaper_worst_bpc']['mean']<0 and report[k]['newspaper_worst_bpc']['ng_better_count']>=2 for k in report)
    out={'protocol_sha256':hashlib.sha256((ROOT/'EQUAL_EPOCH_PROTOCOL.md').read_bytes()).hexdigest(),
        'secondary_exposure_parity_gate_pass':passed,'results':report}
    phase.write_json(ROOT/'EQUAL_EPOCH_RESULTS.json',out)
    print('EQUAL_EPOCH_GATE',passed)
    for k,x in report.items():print(k,x['newspaper_worst_bpc'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('cmd',choices=['fit','analyze']);parser.add_argument('start',type=int,nargs='?',default=0);parser.add_argument('stop',type=int,nargs='?',default=12)
    args=parser.parse_args()
    if args.cmd=='fit':fit(args.start,args.stop)
    else:analyze()
