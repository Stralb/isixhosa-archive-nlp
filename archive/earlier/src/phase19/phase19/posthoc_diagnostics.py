"""Strictly retrospective diagnostics after the frozen Phase 19 primary evaluation.
Not confirmatory and never used to retune the Phase19 policies.
"""
from pathlib import Path
import json,collections
P=Path(__file__).resolve().parent
rows=[json.loads(x) for x in (P/'phase19_per_query.jsonl').open()]
source_stats={}
for ss in ('IMVO','ISIG'):
 z=[r for r in rows if r['source']==ss]
 source_stats[ss]={
  'n':len(z),
  'bm25_hit_and_fuzzy_miss':sum(r['raw_bm25']['top1'] and not r['raw_fuzzy']['top1'] for r in z),
  'fuzzy_hit_and_bm25_miss':sum(r['raw_fuzzy']['top1'] and not r['raw_bm25']['top1'] for r in z),
  'both_hit':sum(r['raw_bm25']['top1'] and r['raw_fuzzy']['top1'] for r in z),
  'both_miss':sum(not r['raw_bm25']['top1'] and not r['raw_fuzzy']['top1'] for r in z),
  'fuzzy_top1_crosses_source':sum(r['raw_fuzzy']['chosen_top1'] is not None and not r['raw_fuzzy']['chosen_top1'].startswith(ss+'_') for r in z),
  'bm25_top1_crosses_source':sum(r['raw_bm25']['chosen_top1'] is not None and not r['raw_bm25']['chosen_top1'].startswith(ss+'_') for r in z),
 }
changed=[]
for r in rows:
 raw=set(r['candidate_pools']['raw_fuzzy']);gated=set(r['candidate_pools']['gated']);en=set(r['candidate_pools']['oracle_english_field']);G=set(r['gold_pages'])
 added=list(gated-raw);removed=list(raw-gated)
 if added:
  changed.append({'qid':r['qid'],'source':r['source'],'added':added,'removed':removed,
                  'new_gold_added':bool(set(added)&G),'old_gold_removed':bool(set(removed)&G),
                  'raw_hit':r['raw_fuzzy']['top1'],'gate_hit':r['gated']['top1']})
out={'posthoc_not_confirmatory':True,'source_rank_inversions':source_stats,
     'lid_predicted_english_on_all':all(not r['predict_xh_view'] for r in rows),
     'gated_equals_unconditional_per_query':all(r['gated']==r['unconditional'] for r in rows),
     'gated_equals_raw_fuzzy_per_query_top1':all(r['gated']['top1']==r['raw_fuzzy']['top1'] for r in rows),
     'gated_candidate_set_different_from_raw_count':len(changed),
     'gated_different_candidate_set_fraction':len(changed)/len(rows),
     'gated_added_new_verbatim_evidence_count':sum(x['new_gold_added'] for x in changed),
     'gated_lost_verbatim_evidence_count':sum(x['old_gold_removed'] for x in changed),
     'candidate_changes':changed,
     'interpretation':'On these English keyword queries the frozen classifier selected English on all queries. An enriched candidate set can only help if raw candidate retrieval missed an annotated evidence page; here the annotated raw Top-20 ceiling was already very high. Full-page partial-ratio reranking often displaced BM25-correct source pages.'}
(P/'posthoc_diagnostics.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!='candidate_changes'},indent=2))
