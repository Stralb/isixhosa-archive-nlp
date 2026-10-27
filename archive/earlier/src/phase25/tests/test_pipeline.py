#!/usr/bin/env python3
"""All scientific labels and small corpora in this test are synthetic fixtures."""
import csv,json,subprocess,sys,tempfile,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src'

def cmd(script,*args,ok=True):
 p=subprocess.run([sys.executable,str(SRC/script),*map(str,args)],capture_output=True,text=True)
 if ok and p.returncode:raise AssertionError((script,p.stdout,p.stderr))
 return p

def write(p,header,records):
 with p.open('w',newline='',encoding='utf-8') as f:
  dw=csv.DictWriter(f,header);dw.writeheader();dw.writerows(records)

def read(p):
 with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

with tempfile.TemporaryDirectory() as dirname:
 T=Path(dirname)
 cmd('create_workbooks.py','--out',T/'templates')
 a=read(T/'templates/query_author_main_80.csv');b=read(T/'templates/query_author_pilot_20.csv')
 assert len(a)==80 and len(b)==20 and not any(x['query_text'] for x in a)
 assert {r['query_id'] for r in a}.isdisjoint({r['query_id'] for r in b})
 assert sum(x['intent']=='evidence' for x in a)==50
 assert sum(x['intent']=='provenance' for x in a)==30
 a=a[:2]
 a[0]['query_text']='Find school records in the newspaper';a[0]['independent_author_attestation']='YES'
 a[1]['query_text']='Find a specific issue from the summer';a[1]['independent_author_attestation']='YES';a[1]['requested_issue_date']='1885-01-01'
 q=T/'queries.csv';write(q,list(a[0]),a)
 assert cmd('validate_queries.py','--queries',q,'--out',T/'checked.json').returncode==0
 a[1]['independent_author_attestation']='NO';write(q,list(a[0]),a)
 assert cmd('validate_queries.py','--queries',q,'--out',T/'bad.json',ok=False).returncode!=0
 a[1]['independent_author_attestation']='YES';write(q,list(a[0]),a)
 pages=T/'pages.jsonl'
 with pages.open('w') as f:
  for i in range(14):
   obj={'id':f'DOC{i:02d}','source':'IMVO' if i<7 else 'ISIG',
     'date':'1885-01-01','raw':('school records children classroom education. '*14 if i%2==0 else 'winter newspaper publication dates issue. '*14)}
   f.write(json.dumps(obj)+'\n')
 cmd('retrieve.py','--queries',q,'--pages',pages,'--out',T/'ranks','--topk','12')
 rows=read(T/'ranks/runs.csv');assert {'BM25_PAGE','BM25_METADATA_DATE','BM25_PASSAGE_MAX','TFIDF_PAGE','RRF_PAGE_PASSAGE_60'}=={r['system'] for r in rows}
 assert len(set((r['system'],r['query_id'],r['rank']) for r in rows))==len(rows)
 assert all(1<=int(r['rank'])<=12 for r in rows)
 cmd('evaluation_toolkit.py','prepare','--queries',q,'--runs',T/'ranks/runs.csv','--out',T/'blind','--depth','10')
 ja=read(T/'blind/judgments_A.csv');jb=read(T/'blind/judgments_B.csv')
 assert len(ja)==len(jb)>0
 assert 'BM25_PAGE' not in (T/'blind/judgments_A.csv').read_text()
 for i,(x,y) in enumerate(zip(ja,jb)):
  x.update({'relevance':'2' if i%2 else '0','provenance_match':'1','verbatim_match':'0','confidence':'high'})
  y.update({'relevance':'2' if i%2 else '0','provenance_match':'1','verbatim_match':'0','confidence':'high'})
 jb[0]['relevance']='1' # a genuine disagreement
 write(T/'blind/judgments_A.csv',list(ja[0]),ja)
 write(T/'blind/judgments_B.csv',list(jb[0]),jb)
 cmd('adjudicate.py','compare','--a',T/'blind/judgments_A.csv','--b',T/'blind/judgments_B.csv','--out',T/'review')
 report=json.loads((T/'review/agreement.json').read_text())
 assert report['needs_adjudication']==1
 assert report['agreement']['relevance']['raw_agreement']<1
 pending=read(T/'review/pending_resolution.csv')
 assert pending[0]['final_relevance']==''
 assert cmd('adjudicate.py','finalize','--resolutions',T/'review/pending_resolution.csv','--out',T/'final','--require-complete',ok=False).returncode!=0
 assert not (T/'final/adjudicated_judgments.csv').exists()
 pending[0]['final_relevance']='2'
 write(T/'review/pending_resolution.csv',list(pending[0]),pending)
 cmd('adjudicate.py','finalize','--resolutions',T/'review/pending_resolution.csv','--out',T/'final','--require-complete')
 qrels=read(T/'final/adjudicated_judgments.csv');assert len(qrels)==len(ja)
 cmd('evaluation_toolkit.py','score','--queries',q,'--runs',T/'ranks/runs.csv','--qrels',T/'final/adjudicated_judgments.csv','--out',T/'scored.json')
 score=json.loads((T/'scored.json').read_text())
 assert all(r['n_judged_top1']>0 for r in score['summaries'])
 cmd('paired_uncertainty.py','--queries',q,'--runs',T/'ranks/runs.csv','--qrels',T/'final/adjudicated_judgments.csv','--out',T/'comparison.json','--boot','100')
 assert json.loads((T/'comparison.json').read_text())['results']
 cmd('build_local_judge_html.py','--pages',pages,'--assessor-sheet',T/'blind/judgments_A.csv','--out',T/'blind/local.html','--max-pairs','2')
 h=(T/'blind/local.html').read_text()
 assert 'school records children' in h and 'BM25_PAGE' not in h and 'Retrieved document ID' not in h
 # Crucial test: blank judgments do not create fabricated accuracy.
 (T/'final/adjudicated_judgments.csv').write_text('query_id,doc_id,relevance,provenance_match,verbatim_match,confidence,adjudication_notes\n')
 assert cmd('evaluation_toolkit.py','score','--queries',q,'--runs',T/'ranks/runs.csv','--qrels',T/'final/adjudicated_judgments.csv','--out',T/'badscore.json',ok=False).returncode!=0
 print('PASS: query completeness guard; 5 real sparse/metadata retrievers; blind pooling; assessor disagreement; strict adjudication; optional unknown guard; scoring/paired CI; local HTML')
