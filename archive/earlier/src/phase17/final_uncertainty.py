from pathlib import Path
from collections import defaultdict
import json,numpy as np
P=Path('/mnt/data/isiXhosa_phase17_deeper_questions')
q=[x for x in map(json.loads,open(P/'language_probes_results.jsonl')) if x['kind']=='5words_English_minor']
raw=[x for x in map(json.loads,open(P/'minority_retrieval_results.jsonl'))]
rows={name:{z['id']:z for z in raw if z['method']==name} for name in ('raw','en','xh')}
methods={}
for name in ('logreg_long','nb_long','logreg_short'):
 p={z['id']:z['pred_xho'] for z in map(json.loads,open(P/('lid_predictions_'+name+'.jsonl'))) if z['kind']=='minority5'}
 methods[name]={z['id']:rows['xh' if p[z['id']] else 'en'][z['id']]['any1'] for z in q}
methods['raw']={z['id']:rows['raw'][z['id']]['any1'] for z in q}
methods['oracle_en']={z['id']:rows['en'][z['id']]['any1'] for z in q}
issues=defaultdict(lambda:defaultdict(list))
for x in q:issues[x['source']][x['issue']].append(x['id'])
rng=np.random.default_rng(20261023)
report={}
for src in ('IMVO','ISIG'):
 ids=[x['id'] for x in q if x['source']==src]
 allissues=list(issues[src])
 pairs=[('logreg_short','logreg_long'),('logreg_short','raw'),('nb_long','logreg_long'),('oracle_en','logreg_short')]
 for a,b in pairs:
  point=float(np.mean([methods[a][id]-methods[b][id] for id in ids]));boot=[]
  for i in range(5000):
   sel=rng.choice(allissues,size=len(allissues),replace=True)
   j=[id for iss in sel for id in issues[src][iss]]
   boot.append(float(np.mean([methods[a][id]-methods[b][id] for id in j])))
  report[f'{src}_{a}_minus_{b}']={'pp':100*point,'ci95_pp':[100*float(v) for v in np.quantile(boot,[.025,.975])],'pages':len(ids),'issues':len(allissues)}
# Look specifically at unique excerpts (one paragraph match source page).
for src in ('IMVO','ISIG'):
 uniq=[x for x in q if x['source']==src and not rows['raw'][x['id']]['multi']]
 for m in methods:
  report[f'{src}_unique_{m}']={'n':len(uniq),'any1':float(np.mean([methods[m][x['id']] for x in uniq])) if uniq else None}
(P/'lid_method_uncertainty.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
# Char-LM comparisons by independent source issues/articles; all 3 seeds paired.
t=[x for x in map(json.loads,open(P/'transfer_charlm_results.jsonl'))]
triples=defaultdict(dict)
for z in t:
 key=(z['source'],z['issue'],z['seed'],z['page']);triples[key][z['method']]=z['bpc']
assert all(len(x)==3 for x in triples.values())
info={}
for src in ('IMVO','ISIG','MODERN'):
 iss=sorted({x[1] for x in triples if x[0]==src})
 for a,b in [('HALF_HALF','HIST_ONLY'),('HALF_HALF','MODERN_ONLY'),('HIST_ONLY','MODERN_ONLY')]:
  by_iss={}
  for issue in iss:
   vals=[mods[a]-mods[b] for key,mods in triples.items() if key[0]==src and key[1]==issue]
   by_iss[issue]=float(np.mean(vals))
  mean=float(np.mean(list(by_iss.values())));boot=[]
  for it in range(5000):
   chosen=rng.choice(iss,size=len(iss),replace=True)
   boot.append(float(np.mean([by_iss[x] for x in chosen])))
  info[f'{src}_{a}_minus_{b}']={'issue_mean_bpc_delta':mean,'ci95':[float(x) for x in np.quantile(boot,[.025,.975])],'issue_groups':len(iss)}
(P/'transfer_issue_uncertainty.json').write_text(json.dumps(info,indent=2))
print('TRANSFER',json.dumps(info,indent=2))
