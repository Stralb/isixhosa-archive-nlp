#!/usr/bin/env python3
"""Compare a new portable run against the first frozen real-data result."""
import json
from pathlib import Path
r=Path(__file__).resolve().parent
old=json.loads((r/'expected_results.json').read_text())
new=json.loads((r/'results.json').read_text())
assert old['phase14_precommit_sha256']==new['phase14_precommit_sha256']
assert old['source_input_sha256']==new['source_input_sha256']
assert old['exhaustive_toy']==new['exhaustive_toy']
cnt=0
for src in ('ityala','zemkinkomo','modern'):
    x=old['sources'][src]; y=new['sources'][src]
    assert x['certificate_summary']==y['certificate_summary']
    assert x['observed_unseen_words']==y['observed_unseen_words']
    assert x['max_edits']==y['max_edits']
    assert x['split_candidates']==y['split_candidates']
    assert x['join_candidates']==y['join_candidates']
    assert len(x['pairs'])==len(y['pairs'])==15
    for a,b in zip(x['pairs'],y['pairs']):
        assert a==b
        cnt+=1
print('PASS:',cnt,'paired comparisons and all error budgets exactly reproduced')
