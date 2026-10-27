from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent
rows=[json.loads(x) for x in (P/'phase19_per_query.jsonl').open()]
r=json.loads((P/'phase19_results.json').read_text())
assert len(rows)==55
assert len({x['issue'] for x in rows})==55
assert r['dataset']['all_source_issues_disjoint_phase18']
assert not r['gate']['H19a_pass']
assert r['frozen_precommit_sha256']==hashlib.sha256((P/'PRECOMMIT.md').read_bytes()).hexdigest()
assert r['frozen_query_sha256']==hashlib.sha256((P/'queries_precommit.jsonl').read_bytes()).hexdigest()
for row in rows:
 assert row['source_page'] in row['gold_pages']
 assert all(len(v)<=20 for v in row['candidate_pools'].values())
 assert row['gated']['top1']==row['raw_fuzzy']['top1']
for method,v in r['metrics']['top1'].items():
 for src in ('IMVO','ISIG'):
  z=[x[method]['top1'] for x in rows if x['source']==src]
  assert abs(sum(z)/len(z)-v['source'][src])<1e-5
print('PASS: frozen hashes, source issues, gold anchors, candidate budgets, saved results and primary rejection.')
