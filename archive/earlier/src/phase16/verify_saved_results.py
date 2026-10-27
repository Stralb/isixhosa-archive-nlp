"""Fast consistency checks; does not rerun retrieval models or resample issues."""
import json
from pathlib import Path
p=Path(__file__).parent/'code'/'isiXhosa_phase15_source_grounded_search'
b=json.loads((p/'paragraph_benchmark_summary.json').read_text())
s=json.loads((p/'paragraph_retrieval_summary.json').read_text())
u=json.loads((p/'issue_grouped_uncertainty.json').read_text())
a=json.loads((p/'archive_language_access_mass.json').read_text())
w=json.loads((p/'within_paragraph_integrity_summary.json').read_text())
assert b['new_query_total']==583
assert b['old_query_audit']['total']==589
assert b['old_query_audit']['in_raw_paragraph']==413
assert b['new_query_repeated_n']==183
assert sum(w[f'{s}_{l}']['total'] for s in ['IMVO','ISIG'] for l in ['xh','en'])==4_939_799
assert abs(s['raw']['clean_all']['any1']-.898799)<.000001
assert abs(s['raw']['noisy_all']['any1']-.720412)<.000001
assert abs(s['fielded_auto']['noisy_all']['any1']-.761578)<.000003, s['fielded_auto']['noisy_all']['any1']
assert abs(u['fielded_auto_minus_raw_noisy_any_top1']['difference_pp']-4.117)<.002
assert abs(a['IMVO']['english_majority_english_retained_frac']-.06647)<1e-4
assert abs(a['ISIG']['english_majority_english_retained_frac']-.207743)<1e-4
print('PASS: all key manuscript numerical values match saved Phase 15 results.')
