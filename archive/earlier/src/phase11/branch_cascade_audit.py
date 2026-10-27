#!/usr/bin/env python3
"""Exploratory AFTER held-out scoring: quantify merge-inventory divergence for the two fitted branches."""
import json, math, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
mod=ROOT/'models'; data=json.loads((ROOT/'PHASE11_RESULTS.json').read_text())
rows=[]
for source,item in data['summary'].items():
 for row in item['rows']:
  ca,cb=row['candidates']
  a=json.loads((mod/ca['file']).read_text())['merges']
  b=json.loads((mod/cb['file']).read_text())['merges']
  va={''.join(p) for p in a};vb={''.join(p) for p in b}
  # recovery of the originally leading pair in the opposite branch
  p=ca['first_pair'];q=cb['first_pair']
  recover_a=next((i for i,z in enumerate(b,1) if z==p),None)
  recover_b=next((i for i,z in enumerate(a,1) if z==q),None)
  rows.append({'source':source,'seed':row['seed'],'first_pairs':[p,q],
         'full_merge_vocabulary_jaccard':len(va&vb)/len(va|vb),
         'n_exclusive_merged_pieces':len(va^vb),
         'original_first_pair_recovery_rank_second_branch':recover_a,
         'second_pair_recovery_rank_greedy_branch':recover_b,
         'top100_merge_symbol_jaccard':len(set(''.join(z) for z in a[:100])&set(''.join(z) for z in b[:100])) / len(set(''.join(z) for z in a[:100])|set(''.join(z) for z in b[:100])),
         'heldout_regret':row['scores']['greedy_regret_E'],
         'heldout_branch_gap_abs':abs(row['scores']['greedy_unseen_E']-row['scores']['second_unseen_E'])})
for source in ['IMVO','ISIG','MODERN']:
 arr=[r for r in rows if r['source']==source]
 print(source,'meanJaccard',round(statistics.mean(x['full_merge_vocabulary_jaccard'] for x in arr),4),
       'range',[round(min(x['full_merge_vocabulary_jaccard'] for x in arr),4),round(max(x['full_merge_vocabulary_jaccard'] for x in arr),4)],
       'median_recovery',statistics.median(x['original_first_pair_recovery_rank_second_branch'] for x in arr if x['original_first_pair_recovery_rank_second_branch'] is not None),
       'exclusive',[min(x['n_exclusive_merged_pieces'] for x in arr),max(x['n_exclusive_merged_pieces'] for x in arr)],
       'mean_abs_gap',round(statistics.mean(x['heldout_branch_gap_abs'] for x in arr),4))
(ROOT/'EXPLORATORY_BRANCH_CASCADE.json').write_text(json.dumps({'posthoc':True,'rows':rows},indent=2,ensure_ascii=False))
