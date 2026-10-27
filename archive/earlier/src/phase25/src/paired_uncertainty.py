#!/usr/bin/env python3
"""Cluster bootstrap paired Top-1 on COMMONLY judged topics. No missing->failure recode."""
import argparse,csv,json,collections
from pathlib import Path
import numpy as np

def load(p):
 with Path(p).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def decode(v):
 v=v.strip()
 return None if v in ('','U','?') else int(v)

def main():
 p=argparse.ArgumentParser();p.add_argument('--queries',required=True);p.add_argument('--runs',required=True)
 p.add_argument('--qrels',required=True);p.add_argument('--out',required=True)
 p.add_argument('--seed',type=int,default=20261025);p.add_argument('--boot',type=int,default=5000)
 args=p.parse_args()
 Q={r['query_id']:r for r in load(args.queries)}
 runs=collections.defaultdict(dict)
 for r in load(args.runs):
  key=(r['system'],r['query_id'])
  if int(r['rank'])==1:
   if key in runs:raise ValueError('duplicate Top-1 run')
   runs[key]=r['doc_id']
 labels={(r['query_id'],r['doc_id']):r for r in load(args.qrels)}
 systems=sorted(set(sys for sys,q in runs))
 results=[]; rng=np.random.default_rng(args.seed)
 for sidx,A in enumerate(systems):
  for B in systems[sidx+1:]:
   for intent in ('evidence','provenance','verbatim_reuse'):
    pairs=collections.defaultdict(list)
    coverage=0;allq=[]
    for qid,q in Q.items():
     if q['intent']!=intent or (A,qid) not in runs or (B,qid) not in runs:continue
     coverage+=1;ra=labels.get((qid,runs[(A,qid)])); rb=labels.get((qid,runs[(B,qid)]))
     if ra is None or rb is None:continue
     key={'evidence':'relevance','provenance':'provenance_match','verbatim_reuse':'verbatim_match'}[intent]
     va=decode(ra[key]);vb=decode(rb[key]);
     if va is None or vb is None:continue
     if intent=='evidence':va=int(va>=1);vb=int(vb>=1)
     pairs[q['topic_id']].append((va,vb)); allq.append(qid)
    n=sum(len(v) for v in pairs.values());cluster=list(pairs)
    if n:
     values=np.asarray([np.mean([a-b for a,b in pairs[t]]) for t in cluster]);counts=np.asarray([len(pairs[t]) for t in cluster])
     delta=float(np.dot(values,counts)/counts.sum())
     if len(cluster)>=2:
      ix=rng.integers(0,len(cluster),size=(args.boot,len(cluster)))
      draw=(values[ix]*counts[ix]).sum(axis=1)/counts[ix].sum(axis=1)
      ci=np.quantile(draw,[.025,.975]).tolist()
     else:ci=None
    else:delta=None;ci=None
    results.append({'system_A':A,'system_B':B,'intent':intent,'n_both_systems_ranked':coverage,
       'n_both_top1_judged':n,'judgment_pair_coverage':n/coverage if coverage else None,
       'n_topics':len(cluster),'A_minus_B_top1':delta,'cluster_bootstrap_CI95':ci,
       'inference_status':'INCOMPLETE: NO ADJUDICATED PAIRS' if not n else ('DESCRIPTIVE ONLY: TOO FEW TOPICS' if len(cluster)<10 else 'DESCRIPTIVE UNDER TOPIC RESAMPLING')})
 Path(args.out).write_text(json.dumps({'warning':'Unjudged queries are excluded, not scored zero. Avoid generalizing from few query authors.',
   'results':results},indent=2))
 print('Paired comparisons:',len(results),'with judged pairs:',sum(r['n_both_top1_judged'] for r in results))

if __name__=='__main__':main()
