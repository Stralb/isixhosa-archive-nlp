#!/usr/bin/env python3
"""Standalone post-hoc structural check, no original text redistributed.
Original ZIP source is user-provided; source SHA is independently verified.
"""
import collections,csv,hashlib,re,sys,zipfile
from pathlib import Path
SOURCE=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[2] / 'data' / 'TxtLangFiles1870-1894_TB.zip'
EXPECTED='375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==EXPECTED
WORD=re.compile(r'[a-z]+')
PATT=re.compile(r'(IMVO|ISIG)_(1885|1888)-\d\d-\d\d_p(\d+)\.txt$')
by=collections.defaultdict(lambda:[0,0])
issues=collections.defaultdict(set)
with zipfile.ZipFile(SOURCE) as z:
 for name in z.namelist():
  if '/XhTxt/' not in name or name.startswith('__MACOSX/'):
   continue
  m=PATT.search(name)
  if not m:continue
  src,year,pos=m.groups()
  words=WORD.findall(z.read(name).decode('utf-8-sig','replace').lower())
  if len(words)<40:continue
  eligible_positions=sum(max(0,len(w)-3) for w in words)
  if not eligible_positions:continue
  key=(src,year,int(pos))
  by[key][0]+=1;by[key][1]+=eligible_positions
  issues[(src,year)].add(name.split('_p')[0].split('/')[-1])
out=Path(__file__).parent/'page_position_eligible_positions.csv'
with out.open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['source','year','position','pages','eligible_positions','mean_per_page'])
 for (src,year,pos),(pages,N) in sorted(by.items()):
  w.writerow([src,year,pos,pages,N,round(N/pages,2)])
for y in ['1885','1888']:
 a=by[('IMVO',y,1)];b=by[('IMVO',y,2)]
 assert a[0]==b[0],(a,b)
 print('Imvo',y,'first-page positions',a[1], 'second-page positions',b[1], 'ratio',round(b[1]/a[1],4),'issues',len(issues[('IMVO',y)]))
 print(' total pages',sum(v[0] for k,v in by.items() if k[:2]==('IMVO',y)))
print('wrote',out)
