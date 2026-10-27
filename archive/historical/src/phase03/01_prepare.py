#!/usr/bin/env python3
"""Construct Phase3 disjoint-year page-frame with development-only word screening.
No masked outcomes or simulation results are evaluated by this script.
"""
import argparse,collections,hashlib,json,re,zipfile
from pathlib import Path
import numpy as np,pandas as pd

EXPECTED='375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5'
RX=re.compile(r'[^\W\d_]+',re.UNICODE)
PATH_RX=re.compile(r'^TxtLangFiles1870-1894_TB/XhTxt/(IMVO|ISIG)_(\d{4}-\d\d-\d\d)_p(\d+)\.txt$')

def sha_file(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def main(source,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 assert sha_file(source)==EXPECTED
 docpaths={}
 with zipfile.ZipFile(source) as z:
  for fn in z.namelist():
   m=PATH_RX.match(fn)
   if m and m.group(2)[:4] in ('1884','1885','1887','1888'):
    s,date,page=m.groups();docpaths.setdefault((s,date),{})[int(page)]=fn
  slots=[]
  for (s,date),entries in sorted(docpaths.items()):
   expected=4 if s=='IMVO' else 8
   if set(entries)!=set(range(1,expected+1)):continue
   split='dev' if int(date[:4])<=1885 else 'test'
   period='A' if date[:4] in ('1884','1887') else 'B'
   for page,fn in sorted(entries.items()):
    blob=z.read(fn)
    tokens=RX.findall(blob.decode('utf-8','replace').casefold())
    assert len(tokens)>0
    slots.append({'split':split,'period':period,'year':int(date[:4]),'source':s,
                  'issue':f'{s}_{date}','date':date,'page':page,'page_half':'front' if page<=expected//2 else 'back',
                  'zip_entry':fn,'text_sha256':hashlib.sha256(blob).hexdigest(),
                  'n_tokens':len(tokens),'tokens':tokens})
 # word selection DEV only
 freq=collections.Counter()
 for x in slots:
  if x['split']=='dev':freq.update(x['tokens'])
 feature=sorted(freq,key=lambda w:(-freq[w],w))[:80]
 assert len(feature)==80 and len(set(feature))==80
 idword={w:j for j,w in enumerate(feature)}
 arr=np.zeros((len(slots),80),dtype=np.int32)
 vocab={}
 for i,s in enumerate(slots):
  cnt=collections.Counter(s['tokens'])
  arr[i,:]=[cnt[w] for w in feature]
  assert arr[i].sum()<=s['n_tokens']
  # stable numeric IDs for rolling 13gram hashes
  for w in s['tokens']:
   if w not in vocab:vocab[w]=len(vocab)+1
 powers=np.array([np.uint64(pow(11400714819323198485,k,1<<64)) for k in range(13)],dtype=np.uint64)
 # issue-local fingerprints, shared across distinct issues within the same split
 for split in ('dev','test'):
  idx=[i for i,s in enumerate(slots) if s['split']==split]
  first_issue={};repeat=set();hashsets={}
  for i in idx:
   s=slots[i];seq=np.fromiter((vocab[w] for w in s['tokens']),dtype=np.uint64,count=s['n_tokens'])
   n=max(0,len(seq)-12)
   h=np.zeros(n,dtype=np.uint64)
   for k in range(13):h+=seq[k:k+n]*powers[k]
   hs=set(map(int,np.unique(h)))
   hashsets[i]=hs
   for hval in hs:
    if hval in first_issue and first_issue[hval]!=s['issue']:repeat.add(hval)
    else:first_issue.setdefault(hval,s['issue'])
  for i in idx:
   s=slots[i];s['shared_13gram_count']=sum(x in repeat for x in hashsets[i]);s['has_cross_issue_13gram']=bool(s['shared_13gram_count'])
  print('CROSS-ISSUE REPRINT FLAG',split,sum(slots[i]['has_cross_issue_13gram'] for i in idx),'/',len(idx),flush=True)
 dev_index=[i for i,s in enumerate(slots) if s['split']=='dev']
 dev_proxy=np.asarray([sum(arr[i,:4])/slots[i]['n_tokens'] for i in dev_index]);avg=float(dev_proxy.mean());std=float(dev_proxy.std())
 assert std>1e-8
 for i,s in enumerate(slots):
  s['month']=int(s['date'][5:7]);s['hidden_text_proxy']=float(np.clip((sum(arr[i,:4])/s['n_tokens']-avg)/std,-2.5,2.5))
 df=pd.DataFrame([{k:v for k,v in s.items() if k!='tokens'} for s in slots]);df.to_csv(out/'frame.csv',index=False)
 np.savez_compressed(out/'counts.npz',counts=arr,n_tokens=df.n_tokens.to_numpy())
 labels=['NULL']*80
 for j in (6,8,10,12,14,16):labels[j]='DOWN'
 for j in (18,24,30,36,42,48):labels[j]='UP'
 pd.DataFrame({'index':range(80),'word_form':feature,'dev_total_count':[freq[w] for w in feature],
               'engineered_status':labels}).to_csv(out/'feature_manifest.csv',index=False)
 shapes=df.groupby(['split','year','source']).agg(issues=('issue','nunique'),pages=('issue','size'),words=('n_tokens','sum'))
 print(shapes.to_string(),flush=True)
 man={'schema':'PRHI_PHASE3_REAL_YEAR_V1','source_zip_sha256':EXPECTED,
      'features_chosen_solely_from_dev':True,'dev_years':[1884,1885],'test_years':[1887,1888],
      'features_sha256':sha_file(out/'feature_manifest.csv'),
      'frame_sha256':sha_file(out/'frame.csv'),'counts_sha256':sha_file(out/'counts.npz'),
      'reprint_fingerprints':'unsigned-64 rolling 13-token hashes; collisions theoretically possible',
      'dev_hidden_proxy_mean':avg,'dev_hidden_proxy_sd':std,
      'shape':{f'{split}_{year}_{source}':{'pages':int(val.pages),'issues':int(val.issues),'words':int(val.words)} for (split,year,source),val in shapes.iterrows()}}
 (out/'frame_manifest.json').write_text(json.dumps(man,indent=2,sort_keys=True)+'\n')
 print('FROZEN',len(slots),'pages, 80 dev-selected word forms',out,flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();main(a.source,a.out)
