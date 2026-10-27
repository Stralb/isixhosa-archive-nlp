import json
from pathlib import Path
r=Path('/mnt/data/isiXhosa_publication_continuation')
meta=json.loads((r/'pilot_charlm_summary.json').read_text())['policies']
policies=sorted(meta)
source=('IMVO','ISIG')
full_macro={m:meta[m]['validation']['macro_bpc'] for m in policies}
margin=0.02
eligible=[m for m in policies if full_macro[m] <= min(full_macro.values())+margin]
by_source={}
for m in policies:
 by_source[m]={}
 for s in source:
  r0=meta[m]['validation'][s]
  total=r0['chars_all'];non=r0['chars_novel'];match=total-non
  assert match>0
  repeat_bpc=(r0['bpc_all']*total - r0['bpc_no_train_repeat']*non)/match
  nonmatch=r0['bpc_no_train_repeat']
  by_source[m][s]={'nonmatch':nonmatch,'match':repeat_bpc,'q_obs':match/total,
                    'max_robust_q_0_to_0.25':max(nonmatch,0.75*nonmatch+0.25*repeat_bpc)}
robust={m:max(by_source[m][s]['max_robust_q_0_to_0.25'] for s in source) for m in eligible}
nonmatch_minimax={m:max(by_source[m][s]['nonmatch'] for s in source) for m in policies}
nonmatch_macro={m:sum(by_source[m][s]['nonmatch'] for s in source)/2 for m in policies}
select=lambda scores:min(scores,key=lambda m:(scores[m],m))
result={'status':'computed exclusively from 1887 validation; no future-year contents used',
 'precommit':'PHASE3_PRECOMMIT.md', 'eligible':eligible,'full_macro':full_macro,
 'models':by_source,'candidate_robust_scores':robust,
 'simple_minimax_nonmatch':nonmatch_minimax,
 'simple_macro_nonmatch':nonmatch_macro,
 'selected':{'robust':select(robust),'full_macro':select(full_macro),'nonmatch_macro':select(nonmatch_macro),'nonmatch_minimax':select(nonmatch_minimax)},
 'robust_equivalence':all(abs(robust[m]-nonmatch_minimax[m])<1e-10 for m in eligible),
 'all_policies_have_match_bpc_below_nonmatch':all(by_source[m][s]['match']<=by_source[m][s]['nonmatch'] for m in policies for s in source)}
assert result['robust_equivalence'] and result['all_policies_have_match_bpc_below_nonmatch']
out=r/'phase3'/'selector_preflight.json'
out.write_text(json.dumps(result,indent=2))
print(json.dumps({'selected':result['selected'],'robust_equivalence':result['robust_equivalence'],
'all_policies_have_match_bpc_below_nonmatch':result['all_policies_have_match_bpc_below_nonmatch'],
'candidate_robust_scores':robust},indent=2))
