from pathlib import Path
import json, numpy as np
from collections import defaultdict
P=Path('/mnt/data/isiXhosa_phase17_deeper_questions')
r=[json.loads(x) for x in open(P/'field_density_results.jsonl')]
rng=np.random.default_rng(20261023);out={}
for source in ('IMVO','ISIG'):
 arr=[x for x in r if x['source']==source]
 groups=defaultdict(list)
 for x in arr:groups[x['issue']].append(x)
 issues=list(groups)
 low=[x for x in arr if x['en_word_count']<50]; high=[x for x in arr if x['en_word_count']>=50]
 point=float(np.mean([x['field_agreement'] for x in high])-np.mean([x['field_agreement'] for x in low]));boot=[]
 for j in range(10000):
  sample=[x for i in rng.choice(issues,size=len(issues),replace=True) for x in groups[i]]
  a=[x['field_agreement'] for x in sample if x['en_word_count']<50];b=[x['field_agreement'] for x in sample if x['en_word_count']>=50]
  if a and b:boot.append(float(np.mean(b)-np.mean(a)))
 out[source]={'low_n':len(low),'low_agree':float(np.mean([x['field_agreement'] for x in low])),'high_n':len(high),'high_agree':float(np.mean([x['field_agreement'] for x in high])),'high_minus_low':point,'issue_boot_ci95':[float(x) for x in np.quantile(boot,[.025,.975])],'issues':len(issues)}
print(json.dumps(out,indent=2))
(P/'density_issue_uncertainty.json').write_text(json.dumps(out,indent=2))
