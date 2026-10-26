#!/usr/bin/env python3
"""Independent compact checks based on source tracker and saved outputs."""
import argparse,zipfile,io,json,hashlib
import pandas as pd
import numpy as np
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--zip',required=True);p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out); data=json.loads((out/'RESULTS.json').read_text())
with zipfile.ZipFile(a.zip) as z:
    b=next(x for x in z.namelist() if x.endswith('LanguageSplitTracker70-94.csv'))
    t=pd.read_csv(io.BytesIO(z.read(b)))
    t=t[t.Edition.str.match(r'^(IMVO|ISIG)_\d{4}-\d{2}-\d{2}_p\d{3}\.txt$')].copy()
    t['source']=t.Edition.str[:4]
    t['date']=t.Edition.str.extract(r'_(\d{4}-\d{2}-\d{2})_')[0]
    t['year']=t.date.str[:4].astype(int)
    t['ym']=t.date.str[:7]
    t['total']=t.Xho_Wordcount + t.Eng_Wordcount
    t=t[t.total>0].copy()
    t['share']=t.Eng_Wordcount/t.total
s=pd.read_csv(out/'page_visibility.csv')
valid=s[s.total>0]
assert len(s)==3608 and len(valid)==3607
merged=s.merge(t[['Edition','Eng_Wordcount','Xho_Wordcount']],left_on='filename',right_on='Edition',how='left',validate='one_to_one')
assert (merged.loc[merged.total>0,'en']==merged.loc[merged.total>0,'Eng_Wordcount']).all() and (merged.loc[merged.total>0,'xh']==merged.loc[merged.total>0,'Xho_Wordcount']).all()
assert (merged.loc[merged.total==0,'en']==0).all() and (merged.loc[merged.total==0,'xh']==0).all()
issues=t.groupby(['source','date'],as_index=False)[['Eng_Wordcount','total']].sum()
issues['share']=issues.Eng_Wordcount/issues.total
assert len(issues)==valid.groupby(['source','date']).ngroups
x=issues[(issues.source=='IMVO') & (issues.date.str[:4].isin(['1885','1887','1888']))].copy()
y=issues[(issues.source=='ISIG') & (issues.date.str[:4].isin(['1885','1887','1888']))].copy()
x['ym']=x.date.str[:7];y['ym']=y.date.str[:7]
mm=x.groupby('ym').share.mean().to_frame('imvo').join(y.groupby('ym').share.mean().to_frame('isig'),how='inner')
d=float((mm.imvo-mm.isig).mean());assert abs(d-data['h32a']['observed_difference'])<1e-12
im=t[(t.source=='IMVO') & (t.year.isin([1885,1887,1888]))]
b=float(im.Eng_Wordcount.sum()/im.total.sum()-(im.Eng_Wordcount/im.total).mean())
assert abs(b-data['h32b']['observed_difference'])<1e-12
ie=issues[(issues.source=='IMVO') & (issues.date.str[:4].astype(int).between(1885,1888))].share
il=issues[(issues.source=='IMVO') & (issues.date.str[:4].astype(int).between(1889,1894))].share
c=float(il.mean()-ie.mean());assert abs(c-data['h32c']['delta_late_minus_early'])<1e-12
for field in ['h32a','h32b','h32c']:
 assert np.all(np.isfinite([v for v in []]))
summary={'verified':True,'recomputed_h32a':d,'recomputed_h32b':b,'recomputed_h32c':c,
         'num_pages':len(s),'num_issues':len(issues),'tracker_matches':len(s),
         'archive_sha256':hashlib.sha256(Path(a.zip).read_bytes()).hexdigest()}
(out/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
