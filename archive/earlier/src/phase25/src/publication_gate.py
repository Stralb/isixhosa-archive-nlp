#!/usr/bin/env python3
"""Report evidence gates; deliberately refuses to declare publication validation without humans."""
import json,argparse,csv
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--queries',required=True);p.add_argument('--runs');p.add_argument('--qrels');p.add_argument('--agreement');p.add_argument('--dense-manifest');p.add_argument('--out',required=True)
a=p.parse_args()
def read(path):
 if not path or not Path(path).is_file():return []
 with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
qs=read(a.queries);runs=read(a.runs);qrels=read(a.qrels)
attested=[q for q in qs if q.get('query_text','').strip() and q.get('independent_author_attestation')=='YES']
authors=set(q.get('author_id','') for q in attested)
refs=[q for q in attested if q.get('known_reference_page_id') and q.get('reference_basis')]
agg=json.loads(Path(a.agreement).read_text()) if a.agreement and Path(a.agreement).is_file() else None
systems=set(x['system'] for x in runs)
qset=set(q['query_id'] for q in qs)
judged=set((r['query_id'],r['doc_id']) for r in qrels)
all_top1={(r['query_id'],r['doc_id']) for r in runs if r['rank']=='1'}
coverage=len(judged&all_top1)/len(all_top1) if all_top1 else 0
checks={
 '>=80_nonempty_self_attested_queries':len(attested)>=80,
 '>=4_distinct_declared_authors':len(authors)>=4,
 '>=20_independently_documented_known_reference_queries_for_single_page_comparison':len(refs)>=20,
 '>=3_fixed_lexical_systems_executed':len(systems & {'BM25_PAGE','BM25_PASSAGE_MAX','TFIDF_PAGE','RRF_PAGE_PASSAGE_60','BM25_METADATA_DATE'})>=3,
 'dense_retrieval_manifest_present':bool(a.dense_manifest and Path(a.dense_manifest).is_file()),
 'two_assessor_agreement_report_present':agg is not None,
 'human_adjudicated_top1_coverage_>=95_percent':coverage>=.95 and bool(all_top1),
 'all_80_queries_have_some_adjudicated_label':len(set(r['query_id'] for r in qrels)&qset)>=80,
 'ethics_permissions_qualified_expertise_verified_externally':False,
 'independent_query_authorship_verified_externally':False}
res={'gate_kind':'publication_readiness_automated_checks','input':{'query_rows':len(qs),'self_attested_nonempty':len(attested),
  'distinct_author_ids':len(authors),'known_reference_count':len(refs),'run_systems':sorted(systems),
  'qrel_rows':len(qrels),'top1_annotation_coverage':coverage},'gates':checks,
 'automated_claim':'BLOCKED' if not all(checks.values()) else 'REQUIRES MANUAL SCIENTIFIC REVIEW',
 'warning':'A passing machine format check cannot certify real human authorship, consent, relevant expertise, or research novelty.'}
Path(a.out).write_text(json.dumps(res,indent=2))
print('GATES:',sum(checks.values()),'of',len(checks),'passed. Claim:',res['automated_claim'])
