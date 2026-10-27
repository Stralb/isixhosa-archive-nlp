#!/usr/bin/env python3
"""Training-only, paragraph-respecting duplicate-annotation feasibility audit.
No claimed historical genre labels or future-text evaluation.
"""
import collections,json,re,zipfile,hashlib,pathlib
ROOT=pathlib.Path('/mnt/data/isiXhosa_publication_continuation/phase6')
ZIP=pathlib.Path('/mnt/data/TxtLangFiles1870-1894_TB.zip')
RX=re.compile(r'^(IMVO|ISIG)_(1887)-(\d\d)-(\d\d)_p(\d+)\.txt$')
RX_WORD=re.compile(r'[^\W\d_]+',re.UNICODE)
N=12

def digest(seq):return hashlib.blake2b(' '.join(seq).encode('utf-8'),digest_size=12).hexdigest()

pars=[]
with zipfile.ZipFile(ZIP) as z:
 for path in z.namelist():
  if '/XhTxt/' not in path or '/._' in path or not path.endswith('.txt'):continue
  name=path.rsplit('/',1)[-1];ma=RX.fullmatch(name)
  if not ma or int(ma.group(3))>9:continue
  src,year,mo,day,slot=ma.groups();txt=z.read(path).decode('utf-8','replace')
  for pi,block in enumerate(re.split(r'\n\s*\n',txt)):
   ts=RX_WORD.findall(block.casefold());
   if len(ts)<N:continue
   pars.append({'source':src,'issue':f'{src}_1887-{mo}-{day}','month':f'1887-{mo}','file':name,'page':int(slot),'paragraph_index':pi,'tokens':ts,'raw':block})
pars.sort(key=lambda x:(x['file'],x['paragraph_index']))
idx=collections.defaultdict(list)
for i,p in enumerate(pars):
 for j in range(len(p['tokens'])-N+1):idx[digest(p['tokens'][j:j+N])].append((i,j))
meta={}
for h,spots in idx.items():
 issues={pars[i]['issue'] for i,pos in spots}
 if len(issues)<2:continue
 months={pars[i]['month'] for i,pos in spots}
 byslot=collections.defaultdict(set)
 for i,j in spots:byslot[pars[i]['page']].add(pars[i]['issue'])
 pers=len(issues)>=3 and len(months)>=2 and max(map(len,byslot.values()))>=3
 meta[h]={'class':'persistent' if pers else 'transient','issues':len(issues),'occurrences':len(spots),'sources':sorted({pars[i]['source'] for i,j in spots})}
perpar=collections.defaultdict(lambda:{'persistent':set(),'transient':set()})
for h,d in meta.items():
 for i,j in idx[h]:perpar[i][d['class']].add(j)
result={'input_zip_sha256':hashlib.sha256(ZIP.read_bytes()).hexdigest(),'minimum_shingle':N,'block_rule':'double-newline paragraphs, no shingles spanning boundaries','paragraphs_ge12':len(pars),'all_shingle_types':len(idx),'repeated_shingle_types':len(meta),'counts':{}}
perpar=dict(perpar)
for src in ['IMVO','ISIG']:
 pool=[(i,p) for i,p in enumerate(pars) if p['source']==src]
 stats={'paragraphs_ge12':len(pool),'issues':len({p['issue'] for i,p in pool})}
 for typ in ('persistent','transient','unmatched'):
  eligible=[(i,p) for i,p in pool if (bool(perpar.get(i,{}).get('persistent')) if typ=='persistent' else bool(perpar.get(i,{}).get('transient')) if typ=='transient' else (i not in perpar))]
  count_page=collections.Counter(p['file'] for _,p in eligible)
  count_issue=collections.Counter(p['issue'] for _,p in eligible)
  stats[typ]={'paragraphs':len(eligible),'source_pages':len(count_page),'issues':len(count_issue),'max_per_issue':max(count_issue.values(),default=0),'max_per_page':max(count_page.values(),default=0),'top_pages':count_page.most_common(4)}
 result['counts'][src]=stats
print(json.dumps(result,indent=2))
ROOT.joinpath('paragraph_feasibility.json').write_text(json.dumps(result,indent=2))
