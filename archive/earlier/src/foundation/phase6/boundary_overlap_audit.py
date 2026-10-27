#!/usr/bin/env python3
"""Retrospective 1887/1888 page-vs-paragraph exact-overlap sensitivity."""
import collections, json, re, sys, zipfile
from pathlib import Path
sys.path.insert(0,'/mnt/data/isiXhosa_publication_continuation')
from audit_overlap import load_pages, digest
ROOT=Path(__file__).resolve().parent
DATA='/mnt/data/TxtLangFiles1870-1894_TB.zip'
RXWORD=re.compile(r'[^\W\d_]+',re.UNICODE)

def paragraph_map(names):
 out={}
 with zipfile.ZipFile(DATA) as z:
  for path in z.namelist():
   fn=path.rsplit('/',1)[-1]
   if fn not in names or '/XhTxt/' not in path or '/._' in path:continue
   text=z.read(path).decode('utf8','replace')
   blocks=[]; allt=[]
   for p in re.split(r'\n\s*\n',text):
    t=RXWORD.findall(p.casefold())
    if t:blocks.append((len(allt),t)); allt+=t
   assert allt==names[fn]['tokens'],fn
   out[fn]=blocks
 return out

def pg_index(train,n=12):
 page_ix=set();para_ix=set()
 for p in train:
  t=p['tokens']
  for i in range(max(0,len(t)-n+1)):
   page_ix.add(digest(t[i:i+n]))
 return page_ix

def para_index(train,bmap,n=12):
 seen=set()
 for p in train:
  for start,toks in bmap[p['name']]:
   for i in range(max(0,len(toks)-n+1)):
    seen.add(digest(toks[i:i+n]))
 return seen

def audit(parts,bmap,pageix,paraix,n=12):
 rows=[]
 for p in parts:
  t=p['tokens']; a=[False]*len(t);b=[False]*len(t)
  for i in range(max(0,len(t)-n+1)):
   h=digest(t[i:i+n])
   if h in pageix:
    for j in range(i,i+n):a[j]=True
  for off,block in bmap[p['name']]:
   for i in range(max(0,len(block)-n+1)):
    h=digest(block[i:i+n])
    if h in paraix:
     for j in range(off+i,off+i+n):b[j]=True
  assert not any(q and not m for q,m in zip(b,a)),p['name']
  rows.append({'source':p['source'],'name':p['name'],'issue':p['issue'],
               'words':len(t),'page_matched':sum(a),'paragraph_matched':sum(b),'extra_due_to_boundary':sum(v and not w for v,w in zip(a,b))})
 result={}
 for src in ('IMVO','ISIG','ALL'):
  r=[row for row in rows if row['source']==src or src=='ALL']
  total=sum(x['words'] for x in r)
  page=sum(x['page_matched'] for x in r)
  para=sum(x['paragraph_matched'] for x in r)
  result[src]={'pages':len(r),'issues':len(set(x['issue'] for x in r)),'words':total,
               'page_concat_fraction':page/total,'paragraph_respecting_fraction':para/total,
               'difference_percentage_points':100*(page-para)/total,
               'relative_decrease_vs_page_concat':(page-para)/page if page else None,
               'pages_with_page_matches':sum(x['page_matched']>0 for x in r),
               'pages_with_paragraph_matches':sum(x['paragraph_matched']>0 for x in r)}
 return result,rows

def main():
 pages=load_pages();lookup={p['name']:p for p in pages}
 bmap=paragraph_map(lookup)
 train=[x for x in pages if x['split']=='train']
 val=[x for x in pages if x['split']=='validation']
 test=[x for x in pages if x['split']=='old_test']
 pageix=pg_index(train);paraix=para_index(train,bmap)
 output={'status':'RETROSPECTIVE_NOT_NEW_CONFIRMATION','n':12,
         'train':{'pages':len(train),'page_shingles':len(pageix),'paragraph_shingles':len(paraix)},
         '1887_validation':audit(val,bmap,pageix,paraix)[0],
         '1888_old_evaluation':audit(test,bmap,pageix,paraix)[0]}
 (ROOT/'boundary_sensitivity_results.json').write_text(json.dumps(output,indent=2))
 print(json.dumps(output,indent=2))
if __name__=='__main__':main()
