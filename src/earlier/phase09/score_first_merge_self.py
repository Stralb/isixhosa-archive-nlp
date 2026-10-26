#!/usr/bin/env python3
"""Training-only selection of initial BPE branch; no held-out text used to decide."""
import collections,hashlib,json,math,numpy as np
from pathlib import Path
import run_phase9 as p
from classic_bpe_intervene import ClassicBPE
ROOT=p.ROOT
assert hashlib.sha256((ROOT/'SELF_SCORE_BRANCH_PROTOCOL.md').read_bytes()).hexdigest()==(ROOT/'SELF_SCORE_BRANCH_PROTOCOL.sha256').read_text().split()[0]
first=json.loads((ROOT/'first_merge_all_seed_results.json').read_text())
phase9=json.loads((ROOT/'results.json').read_text())
existing={(r['mix'],r['seed']):r for r in phase9['runs']}
records=first['records']
models={(r['seed'],r['treatment']):ClassicBPE.from_json(json.loads((ROOT/'first_merge_models'/f'draw{r["seed"]}_first_{r["treatment"]}.json').read_text())) for r in records}

chosen=[]
for mix in (.5,1.):
 for seed in p.SEEDS:
    wordlist=p.get_sample(mix,seed); cs=collections.Counter(wordlist)
    pc=collections.Counter()
    for w,n in cs.items():
        t=['▁',*w]
        for pair,multiplicity in collections.Counter(zip(t,t[1:])).items(): pc[pair]+=n*multiplicity
    top=sorted(pc.items(),key=lambda z:(-z[1],z[0]))[:2]
    margin=(top[0][1]-top[1][1])/top[0][1]
    natural=ClassicBPE.from_json(json.loads((p.MODELS/f'classic_mix{int(mix*100):03d}_draw{seed}.json').read_text()))
    assert natural.merges[0]==top[0][0]
    assert top[0][1] > 0
    treat={'mix':mix,'seed':seed,'top_pair':list(top[0][0]),'second_pair':list(top[1][0]),
           'top_count':top[0][1],'second_count':top[1][1],'relative_margin':margin,'trigger':margin<=.02}
    if margin<=.02:
        assert mix==1.,'unexpected balanced trigger; newly train branch and record'
        assert {tuple(x[0]) for x in top}=={('n','g'),('▁','n')}
        a=models[(seed,'ng')];b=models[(seed,'word_start_n')]
        def selfcost(m):
            return 100*sum(n*m.encode_len(w) for w,n in cs.items())/p.p8.BUDGET
        aE=selfcost(a);bE=selfcost(b)
        selected='ng' if aE<bE else 'word_start_n' if bE<aE else 'ng' # lex tie default
        E=next(r['worst_unseen_E'] for r in records if r['seed']==seed and r['treatment']==selected)
        treat.update({'ng_training_E':aE,'wordstart_training_E':bE,'chosen_first_branch':selected,
           'selected_val_worst_unseen_E':E,'selected_training_E':min(aE,bE)})
    else:
        treat.update({'chosen_first_branch':'ordinary_greedy',
            'selected_val_worst_unseen_E':existing[(mix,seed)]['validation']['worst_unseen_E']})
    treat.update({'natural_val_worst_unseen_E':existing[(mix,seed)]['validation']['worst_unseen_E'],
         'benefit_over_natural':existing[(mix,seed)]['validation']['worst_unseen_E']-treat['selected_val_worst_unseen_E']})
    chosen.append(treat)
    print('TRAIN_RULE',mix,seed,'margin',round(margin,4),'trigger',treat['trigger'],'choose',treat['chosen_first_branch'],
          'benefit',round(treat['benefit_over_natural'],4))

report={};rng=np.random.default_rng(20261023);B=10000
for mix in (.5,1.):
    v=[r for r in chosen if r['mix']==mix]
    ref=np.array([r['natural_val_worst_unseen_E'] for r in v]);sel=np.array([r['selected_val_worst_unseen_E'] for r in v]);improve=ref-sel
    ind=rng.integers(0,20,size=(B,20))
    ci=np.quantile(improve[ind].mean(axis=1),[.025,.975]).tolist()
    report[f'mix_{int(mix*100)}']={'triggered':sum(r['trigger'] for r in v),
       'chosen_first_branches':dict(collections.Counter(r['chosen_first_branch'] for r in v)),
       'baseline_mean':float(ref.mean()),'baseline_std':float(ref.std(ddof=1)),
       'selected_mean':float(sel.mean()),'selected_std':float(sel.std(ddof=1)),
       'baseline_q90':float(np.quantile(ref,.9)),'selected_q90':float(np.quantile(sel,.9)),
       'mean_improvement':float(improve.mean()),'improvement_bootstrap_95':ci,
       'improved':int(sum(improve>1e-7)),'harmed':int(sum(improve< -1e-7))}
report['heuristic_gate_pass']=(report['mix_100']['chosen_first_branches'].get('ng',0)>=18 and report['mix_100']['improvement_bootstrap_95'][0]>0)
(ROOT/'self_score_branch_results.json').write_text(json.dumps({'status':'posthoc fixed threshold on previously inspected data; not novel or confirmatory',
     'protocol_sha256':hashlib.sha256((ROOT/'SELF_SCORE_BRANCH_PROTOCOL.md').read_bytes()).hexdigest(),
     'results':report,'runs':chosen},indent=2,ensure_ascii=False))
print('SUMMARY',json.dumps(report,indent=2))
