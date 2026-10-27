import json, numpy as np
from pathlib import Path
P=Path('/mnt/data/isiXhosa_phase17_deeper_questions')
Q=[json.loads(x) for x in open(P/'language_probes_results.jsonl') if '5words_English_minor' in x]
minor=[q for q in Q if q.get('kind')=='5words_English_minor']
assert len(minor)==466
rawrows=[json.loads(x) for x in open(P/'minority_retrieval_results.jsonl')]
idx={m:{x['id']:x for x in rawrows if x['method']==m} for m in ('raw','en','xh')}
scores={}
for m in ('logreg_long','nb_long','logreg_short'):
 d={x['id']:x['pred_xho'] for x in map(json.loads,open(P/('lid_predictions_'+m+'.jsonl'))) if x['kind']=='minority5'}
 results={}
 for source in ('IMVO','ISIG'):
  query=[x for x in minor if x['source']==source]
  top1=np.mean([idx['xh' if d[x['id']] else 'en'][x['id']]['any1'] for x in query]);top5=np.mean([idx['xh' if d[x['id']] else 'en'][x['id']]['any5'] for x in query]);acc=np.mean([1-d[x['id']] for x in query]);results[source]={'n':len(query),'any1':float(top1),'any5':float(top5),'lid_acc':float(acc)}
 scores[m]=results
(P/'lid_retrieval_controls_summary.json').write_text(json.dumps(scores,indent=2))
print(json.dumps(scores,indent=2))
