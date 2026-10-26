#!/usr/bin/env python3
"""Safely merge scored lexical and optional dense run CSVs (never merge qrels)."""
import argparse,csv
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--out',required=True)
a=p.parse_args();out=[];keys=set();seen={}
for src in a.inputs:
 with Path(src).open(newline='',encoding='utf-8-sig') as f:
  rows=list(csv.DictReader(f))
 for r in rows:
  k=(r['system'],r['query_id'],int(r['rank']))
  if k in keys:raise SystemExit(f'Duplicate ranking: {k}')
  keys.add(k);out.append(r)
  pkey=(r['system'],r['query_id']);seen.setdefault(pkey,[]).append((int(r['rank']),r['doc_id']))
for k,v in seen.items():
 if sorted(rank for rank,d in v)!=list(range(1,len(v)+1)):raise SystemExit(f'Non-consecutive ranks: {k}')
 if len(set(d for rank,d in v))!=len(v):raise SystemExit(f'Duplicate doc in {k}')
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
with Path(a.out).open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,['system','query_id','doc_id','rank','score']);w.writeheader();w.writerows(out)
print('Merged',len(out),'rank entries from',len(a.inputs),'files for',len(seen),'system/query pairs')
