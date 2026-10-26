import zipfile, re, json, csv, io
from pathlib import Path
from collections import Counter, defaultdict
BASE=Path(__file__).resolve().parent; RAW=BASE/'source_data/1. TxtFiles1870-1894_TB.zip'; VIEWS=BASE/'source_data/TxtLangFiles1870-1894_TB.zip'
PAT=re.compile(r'(IMVO|ISIG)_\d{4}-\d\d-\d\d_p\d{3}\.txt$',re.I)
with zipfile.ZipFile(RAW) as z:
 raw={v.split('/')[-1]:z.read(v).decode('utf-8-sig','replace').replace('\r\n','\n') for v in z.namelist() if '/TxtFiles1870-1894_TB/' in v and PAT.search(v) and '__MACOSX' not in v}
with zipfile.ZipFile(VIEWS) as z:
 views={}
 for lg in ('EnTxt','XhTxt'):
  views[lg]={v.split('/')[-1]:z.read(v).decode('utf-8-sig','replace').replace('\r\n','\n') for v in z.namelist() if '/'+lg+'/' in v and PAT.search(v) and '__MACOSX' not in v}
assert len(raw)==len(views['EnTxt'])==len(views['XhTxt'])==3608
pages=[]
for fname in sorted(raw):
 s,date,page=fname[:-4].split('_')
 pages.append(dict(id=fname,source=s,year=int(date[:4]),date=date,issue=s+'_'+date,page=page,raw=raw[fname],en=views['EnTxt'][fname],xh=views['XhTxt'][fname]))
with (BASE/'pages.jsonl').open('w') as f:
 for r in pages:f.write(json.dumps(r,ensure_ascii=False)+'\n')
for period,sel in [('imvo_1890+',lambda p:p['source']=='IMVO' and p['year']>=1890),('isig_1870-75',lambda p:p['source']=='ISIG' and p['year']<=1875),('isig_1876+',lambda p:p['source']=='ISIG' and p['year']>=1876)]:
 cand=[p for p in pages if sel(p)];eligible=[p for p in cand if len(p['en'].split())>=100];paragraphs=[(p,pp) for p in eligible for pp in re.split(r'\n\s*\n',p['en']) if len(pp.split())>=50]
 print(period,'pages',len(cand),'english_100+_pages',len(eligible),'eligible_paras',len(paragraphs),'issues',len(set(p['issue'] for p in eligible)))
print('WROTE',BASE/'pages.jsonl')
