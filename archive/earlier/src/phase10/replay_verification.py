#!/usr/bin/env python3
"""Round-trip verification of two saved neural model checkpoints."""
import json,math
import torch
from pathlib import Path
import neural_merge_test as n
ROOT=n.ROOT
torch.set_num_threads(2)
report=[]
for prefix,name,seed in [('run','imvo5004_second',101),('epoch','imvo5004_second',101)]:
    saved=json.loads((ROOT/f'{prefix}_{name}_{seed}.json').read_text())
    v=json.loads((ROOT/'PREPARED_MANIFEST.json').read_text())['models'][name]['vocab']
    model=n.FFTokenLM(v)
    checkpoint_name=(f'epoch_{name}_{seed}.pt' if prefix=='epoch' else f'{name}_seed{seed}.pt')
    state=torch.load(ROOT/'neural_models'/checkpoint_name,map_location='cpu',weights_only=True)
    model.load_state_dict(state);model.eval()
    diffs={}
    for corpus in ('IMVO','ISIG','ityala','zem','modern'):
        X,Y=n.prepare_tensors(name,corpus)
        out=n.evaluate(model,X,Y,ROOT/'cache'/f'{name}_{corpus}.json')
        error=abs(out['bpc']-saved['evaluation'][corpus]['bpc'])
        assert error<1e-8,(prefix,name,corpus,error)
        diffs[corpus]=error
    report.append({'model':name,'fit':prefix,'seed':seed,'max_abs_BPC_error':max(diffs.values()),'passed':True})
(ROOT/'NEURAL_REPLAY_VERIFICATION.json').write_text(json.dumps(report,indent=2))
print(report)
