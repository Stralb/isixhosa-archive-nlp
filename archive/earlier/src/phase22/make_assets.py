"""Deterministic post-result tables/figures; does not modify the frozen evaluation."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).parent
S=json.loads((P/'phase22_results.json').read_text())
Q=[json.loads(l) for l in (P/'paired_queries_frozen.jsonl').open()]
R=[json.loads(l) for l in (P/'per_query.jsonl').open()]
methods=S['methods']
with (P/'phase22_summary.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['method','external_top1','edited_top1','external_recall20','edited_recall20','top1_delta_pp','recall20_delta_pp'])
 for m in methods:
  x=S['metric_by_mode']['external'][m];y=S['metric_by_mode']['clean'][m]
  w.writerow([m,x['top1'],y['top1'],x['recall20'],y['recall20'],100*(x['top1']-y['top1']),100*(x['recall20']-y['recall20'])])
with (P/'phase22_alignment_audit.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['source','page_id','url','prior_origin_phases','alignment_ratio','normalized_edit_distance','exact_reprint_pages','bm25_external_recall20','bm25_edited_recall20','bm25_external_top1','bm25_edited_top1'])
 for i,q in enumerate(Q):
  d=S['alignment_and_duplicate_diagnostics'][i]
  a=R[2*i]['methods']['BM25'];b=R[2*i+1]['methods']['BM25']
  w.writerow([q['source'],q['page_id'],q['url'],';'.join(str(x) for x in q['prior_query_origin_phases']),q['alignment_ratio_0_100'],d['query_external_vs_aligned_clean_normalized_levenshtein'],d['n_archive_pages_containing_same_normalized_clean_query'],a['recall20'],b['recall20'],a['top1'],b['top1']])
fig,ax=plt.subplots(figsize=(9.1,4.25))
xx=np.arange(len(methods));w=.36
external=[S['metric_by_mode']['external'][m]['top1']*100 for m in methods]
edited=[S['metric_by_mode']['clean'][m]['top1']*100 for m in methods]
ax.bar(xx-w/2,edited,width=w,label='Corrected archive')
ax.bar(xx+w/2,external,width=w,label='External transcript')
ax.set_xticks(xx);ax.set_xticklabels(['BM25','BM25+fuzzy','Char3','Char3+fuzzy','Xh BM25+fuzzy'],rotation=10,ha='right')
ax.set_ylim(0,100);ax.set_ylabel('Correct originating page at Top-1 (%)')
ax.set_title('Phase 22: matched external vs corrected queries (8 pages)')
ax.legend(loc='upper right');ax.grid(axis='y',alpha=.2);fig.tight_layout()
fig.savefig(P/'phase22_top1.png',dpi=165);plt.close(fig)
print('wrote summary.csv, alignment_audit.csv, top1.png')
