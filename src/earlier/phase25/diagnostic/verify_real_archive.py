#!/usr/bin/env python3
"""Real archive diagnostic ONLY. Point to the original Phase24 package and pages.jsonl.

Defaults work when this Phase25 directory is adjacent to isiXhosa_continuation.
No semantic judgments, independent queries, or novel findings implied.
"""
import argparse,csv,json
from pathlib import Path
from collections import defaultdict
D=Path(__file__).resolve().parent
ROOT=D.parents[1]/'isiXhosa_continuation'
p=argparse.ArgumentParser();p.add_argument('--phase24-per-query',default=str(ROOT/'phase24/per_query.jsonl'))
p.add_argument('--pages',default=str(ROOT/'pages.jsonl'));a=p.parse_args()
Q=list(csv.DictReader((D/'DIAGNOSTIC_SOURCE_DERIVED_QUERIES.csv').open()))
RUNS=list(csv.DictReader((D/'retrieval/runs.csv').open()))
old=[json.loads(x) for x in Path(a.phase24_per_query).open()]
by=defaultdict(list)
for x in RUNS:by[(x['system'],x['query_id'])].append(x)
expected={'BM25_PAGE','BM25_METADATA_DATE','BM25_PASSAGE_MAX','TFIDF_PAGE','RRF_PAGE_PASSAGE_60'}
assert len(Q)==8 and {s for s,k in by}==expected
assert all(len(v)==20 for v in by.values())
matched=0
for rec in Q:
 oldrec=next(x for x in old if x['query']==rec['query_text'] and x['page_id']==rec['diagnostic_origin_page'])
 actual=[x['doc_id'] for x in by[('BM25_PAGE',rec['query_id'])]]
 assert actual==oldrec['methods']['BM25']['ranked_top20']
 matched+=1
 assert [x['doc_id'] for x in by[('BM25_METADATA_DATE',rec['query_id'])]]==actual
pageids={json.loads(x)['id'] for x in Path(a.pages).open()}
assert all(x['doc_id'] in pageids for x in RUNS)
assert all(x['independent_author_attestation']=='NO' for x in Q)
result={'verification':'PASS','status':'ENGINEERING_ONLY_NOT_SCIENTIFIC_RESULTS',
 'corpus_pages':len(pageids),'queries':len(Q),'fixed_methods':sorted(expected),
 'ranked_rows':len(RUNS),'reproduced_original_phase24_BM25_top20_exactly':matched,
 'query_authorship':'ARCHIVE_DERIVED_NOT_HUMAN','human_judgments':0,'model_dense_executed':False}
(D/'REAL_ARCHIVE_VERIFICATION.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
