#!/usr/bin/env python3
"""Exploratory check for nonindependence of 12-word shingle annotation anchors.
For diagnostic only; prospective phase6 sampling protocol must be frozen separately.
"""
import sys,collections,json,pathlib
sys.path.insert(0,'/mnt/data/isiXhosa_publication_continuation')
sys.path.insert(0,'/mnt/data/isiXhosa_publication_continuation/phase5')
from audit_overlap import load_pages,digest
from selective_neural import shingle_occurrences
ROOT=pathlib.Path('/mnt/data/isiXhosa_publication_continuation/phase5')
q=[json.loads(s) for s in (ROOT/'annotation_queue_blind.jsonl').read_text().splitlines()]
k={x['annotation_id']:x for x in map(json.loads,(ROOT/'annotation_heuristic_keys.jsonl').read_text().splitlines())}
pages=[x for x in load_pages() if x['split']=='train']
desc=shingle_occurrences(pages)
parent={h:h for h in desc}; rank={h:0 for h in desc}
def root(x):
 while parent[x]!=x:
  parent[x]=parent[parent[x]];x=parent[x]
 return x
def union(a,b):
 a,b=root(a),root(b)
 if a==b:return
 if rank[a]<rank[b]: a,b=b,a
 parent[b]=a
 if rank[a]==rank[b]:rank[a]+=1
# Intervals overlap when starts differ <=11; join nearby windows occurring in same page.
page_anchor=collections.defaultdict(list)
for p in pages:
 prevh=None;prevpos=-999
 for pos in range(len(p['tokens'])-11):
  h=digest(p['tokens'][pos:pos+12])
  if h not in desc: continue
  if prevh is not None and pos-prevpos<12:
   union(prevh,h)
  prevpos=pos;prevh=h
  page_anchor[p['name']].append((pos,h))
component=collections.defaultdict(lambda:{'shingles':set(),'pages':set(),'issues':set(),'sources':set(),'persist_shingles':0,'trans_shingles':0,'occurrences':0})
for h,(count,persistent) in desc.items():
 d=component[root(h)];d['shingles'].add(h);d['persist_shingles']+=persistent;d['trans_shingles']+=not persistent
for p in pages:
 for pos,h in page_anchor[p['name']]:
  d=component[root(h)];d['pages'].add(p['name']);d['issues'].add(p['issue']);d['sources'].add(p['source']);d['occurrences']+=1
families=component
source_counts=collections.defaultdict(list)
for d in families.values():
 for source in d['sources']:source_counts[source].append(d)
print('all_repeat_shingles',len(desc),'family_count',len(families))
for src in ['IMVO','ISIG']:
 bins=collections.Counter(('persistent' if d['persist_shingles'] else 'transient') for d in source_counts[src])
 print(src,'source_families',len(source_counts[src]),'bins',dict(bins),'top_sizes', sorted([len(d['shingles']) for d in source_counts[src]],reverse=True)[:10])
qfamilies=collections.defaultdict(list)
for rec in q:
 h=bytes.fromhex(k[rec['annotation_id']]['first_file'].split('_')[0]) if False else None
 # lookup anchor by filename and position, avoiding truncated hash collisions
 anchor=(k[rec['annotation_id']]['word_anchor'],rec['page_file']); h = next((hash for pos,hash in page_anchor[rec['page_file']] if pos==anchor[0]),None)
 if h is None:print('missing',rec['annotation_id']);continue
 qfamilies[root(h)].append(rec)
print('queue_family_count',len(qfamilies))
for src in ['IMVO','ISIG']:
 subset=[r for r in q if r['source']==src]
 fam=collections.defaultdict(list)
 for r in subset:
  h=next((hash for pos,hash in page_anchor[r['page_file']] if pos==k[r['annotation_id']]['word_anchor']),None)
  if h:fam[root(h)].append(r)
 print('SAMPLE',src,'n',len(subset),'independent_families',len(fam),'largest_clusters',sorted([len(x) for x in fam.values()],reverse=True)[:10])
 for typ in ('persistent','transient'):
  items=[r for r in subset if k[r['annotation_id']]['heuristic_type']==typ]
  xs=collections.defaultdict(list)
  for r in items:
   h=next((hash for pos,hash in page_anchor[r['page_file']] if pos==k[r['annotation_id']]['word_anchor']),None)
   if h:xs[root(h)].append(r)
  print('  ',typ,'n',len(items),'families',len(xs),'largest',sorted([len(x) for x in xs.values()],reverse=True)[:7])
# show top family examples from ISIG queue for diagnosis
for family,records in sorted(qfamilies.items(),key=lambda x:len(x[1]),reverse=True)[:8]:
 d=families[family];print('GROUP n=',len(records),'src',set(r['source'] for r in records),'unique_pages',len(d['pages']),'issues',len(d['issues']),'persist',d['persist_shingles'],'shingles',len(d['shingles']),'examples',[r['target_text'][:65] for r in records[:2]])
