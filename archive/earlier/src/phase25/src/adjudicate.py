#!/usr/bin/env python3
"""Make an assessor-disagreement report; NEVER auto-resolve disagreement or unknowns.

Call `compare`, ask independent adjudicator to complete the `final_*` columns
in pending_resolution.csv, then `finalize` to create qrels for evaluation.
"""
import argparse,csv,json,math
from pathlib import Path
from collections import Counter
FIELDS=('relevance','provenance_match','verbatim_match')
VALID={'relevance':{'0','1','2'},'provenance_match':{'0','1'},'verbatim_match':{'0','1'}}

def readcsv(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def writecsv(path,rows,cols):
 with Path(path).open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)

def norm(v):
 val=(v or '').strip().upper()
 return val if val not in ('','?','U') else 'U'

def table_of(rows):
 out={}
 for r in rows:
  key=(r['query_id'].strip(),r['candidate_id'].strip())
  if key in out:raise ValueError(f'Duplicate adjudication input key {key}')
  out[key]=r
 return out

def kappa(a,b,weights=False):
 if not a:return None
 labels=sorted(set(a+b))
 if len(labels)<=1:return None
 n=len(a)
 observed=0.0
 ca=Counter(a);cb=Counter(b)
 if weights:
  vals={x:int(x) for x in labels}
  denom=max(1,(max(vals.values())-min(vals.values()))**2)
  weight=lambda x,y:((vals[x]-vals[y])**2)/denom
 else:weight=lambda x,y:float(x!=y)
 for x,y in zip(a,b):observed+=weight(x,y)/n
 expected=sum(weight(x,y)*ca[x]*cb[y]/(n*n) for x in labels for y in labels)
 return float(1-observed/expected) if expected>0 else None

def compare(args):
 A=table_of(readcsv(args.a));B=table_of(readcsv(args.b))
 if set(A)!=set(B):raise ValueError('Assessor sheets cover different query/document candidates')
 rows=[];agreement={};conflicts=0
 for field in FIELDS:
  good=[(norm(A[k][field]),norm(B[k][field])) for k in A if norm(A[k][field])!='U' and norm(B[k][field])!='U']
  a=[x for x,y in good];b=[y for x,y in good]
  agreement[field]={'both_decided':len(good),'n_total_pairs':len(A),
    'raw_agreement':sum(x==y for x,y in good)/len(good) if good else None,
    'cohen_kappa':kappa(a,b,weights=(field=='relevance')),
    'kappa_kind':'quadratic_weighted' if field=='relevance' else 'unweighted'}
 for k in sorted(A):
  aa=A[k];bb=B[k]
  if aa['document_id']!=bb['document_id']:raise ValueError('Candidate document IDs differ between assessor sheets')
  r={'query_id':k[0],'candidate_id':k[1],'doc_id':aa['document_id'],
     'assessor_A_notes':aa.get('adjudication_notes',''),
     'assessor_B_notes':bb.get('adjudication_notes','')}
  pending=False
  for field in FIELDS:
   va=norm(aa.get(field));vb=norm(bb.get(field))
   if va not in VALID[field]|{'U'} or vb not in VALID[field]|{'U'}:raise ValueError(f'Invalid {field}: {va} {vb}')
   r['A_'+field]=va;r['B_'+field]=vb
   r['final_'+field]=va if va==vb and va!='U' else ''
   pending |= not bool(r['final_'+field])
  r['needs_adjudication']='YES' if pending else 'NO'
  conflicts+=pending
  rows.append(r)
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 cols=['query_id','candidate_id','doc_id','needs_adjudication']+list(sum(([f'A_{f}',f'B_{f}',f'final_{f}'] for f in FIELDS),[]))+['assessor_A_notes','assessor_B_notes']
 writecsv(out/'pending_resolution.csv',rows,cols)
 report={'judgment_pairs':len(rows),'needs_adjudication':conflicts,
         'no_scores_until_human_resolution':True,'agreement':agreement}
 (out/'agreement.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2))

def finalize(args):
 rows=readcsv(args.resolutions)
 fields=['query_id','doc_id','relevance','provenance_match','verbatim_match','confidence','adjudication_notes']
 output=[];unresolved=[]
 for r in rows:
  q=r['query_id'];d=r['doc_id'];label={f:norm(r['final_'+f]) for f in FIELDS}
  bad=[f for f,v in label.items() if v=='U' or v not in VALID[f]]
  if bad:
   unresolved.append((q,d,bad));continue
  output.append({'query_id':q,'doc_id':d,'relevance':label['relevance'],
   'provenance_match':label['provenance_match'],'verbatim_match':label['verbatim_match'],
   'confidence':'adjudicated','adjudication_notes':'resolved from two independent assessor sheets'})
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 if unresolved and args.require_complete:
  (out/'adjudication_status.json').write_text(json.dumps({'total':len(rows),'judged':len(output),'unresolved':len(unresolved),'publication_scoring_BLOCKED':True},indent=2))
  raise SystemExit('Incomplete judgments. No final qrels generated in strict mode')
 writecsv(out/'adjudicated_judgments.csv',output,fields)
 (out/'adjudication_status.json').write_text(json.dumps({'total':len(rows),'judged':len(output),
        'unresolved':len(unresolved),'unresolved_qid_doc_field':unresolved,
        'note':'Unresolved candidates are not assigned 0 relevance'},indent=2))
 print(f'Adjudicated: {len(output)} completed; {len(unresolved)} unresolved, excluded as unjudged')
 if unresolved and args.require_complete:raise SystemExit('Incomplete judgments. No final publication scoring permitted')

def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
 a=sub.add_parser('compare');a.add_argument('--a',required=True);a.add_argument('--b',required=True);a.add_argument('--out',required=True);a.set_defaults(func=compare)
 b=sub.add_parser('finalize');b.add_argument('--resolutions',required=True);b.add_argument('--out',required=True);b.add_argument('--require-complete',action='store_true');b.set_defaults(func=finalize)
 x=p.parse_args();x.func(x)
if __name__=='__main__':main()
