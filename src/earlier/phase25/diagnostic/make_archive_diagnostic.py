"""Create an archive-derived *engineering* fixture; NEVER a human evaluation query set."""
import csv,json,argparse
from pathlib import Path
D=Path(__file__).resolve().parent
root=D.parents[1]/'isiXhosa_continuation'
p=argparse.ArgumentParser();p.add_argument('--phase24-sample',default=str(root/'phase24/sample_frozen.jsonl'))
a=p.parse_args()
items=[json.loads(x) for x in Path(a.phase24_sample).open()]
sel=[x for x in items if x['source']=='IMVO'][:4]+[x for x in items if x['source']=='ISIG'][:4]
with (D/'DIAGNOSTIC_SOURCE_DERIVED_QUERIES.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['query_id','query_text','source_scope','intent','language','author_id','topic_id','independent_author_attestation','diagnostic_origin_page']);w.writeheader()
 for i,x in enumerate(sel):
  w.writerow({'query_id':f'DRY{i+1:02d}','query_text':x['query'],'source_scope':x['source'],
    'intent':'evidence','language':'mixed','author_id':'ARCHIVE_EXCERPT_NOT_HUMAN','topic_id':f'DRY{i+1:02d}',
    'independent_author_attestation':'NO','diagnostic_origin_page':x['page_id']})
print('8 archived source-derived fixtures; not human-authored')
