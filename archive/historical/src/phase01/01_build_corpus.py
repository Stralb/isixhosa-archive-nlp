#!/usr/bin/env python3
"""Make a reproducible *word-form only* 1885 negative-control archive table.

No semantic labels or external texts are invented. Source ZIP stays unmodified.
"""
import argparse,collections,csv,hashlib,json,re,zipfile
from pathlib import Path
import numpy as np

RX_PATH=re.compile(r'/XhTxt/(IMVO|ISIG)_(1885-\d{2}-\d{2})_p(\d{3})\.txt$')
RX_WORD=re.compile(r'[^\W\d_]+',re.UNICODE)


def checksum(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
 return h.hexdigest()


def run(zip_path,output):
 output=Path(output);output.mkdir(parents=True,exist_ok=True)
 pages=[];source_counts={src:collections.Counter() for src in ('IMVO','ISIG')}
 total=collections.Counter()
 with zipfile.ZipFile(zip_path) as z:
  for filename in sorted(z.namelist()):
   m=RX_PATH.search(filename)
   if not m:continue
   src,date,pageid=m.group(1),m.group(2),int(m.group(3))
   raw=z.read(filename)
   words=RX_WORD.findall(raw.decode('utf-8',errors='replace').casefold())
   count=collections.Counter(words)
   source_counts[src].update(count);total.update(count)
   pages.append(dict(source=src,issue=f'{src}_{date}',date=date,page=pageid,
                     page_half=('front' if pageid<=(2 if src=='IMVO' else 4) else 'back'),
                     zip_entry=filename,text_sha256=hashlib.sha256(raw).hexdigest(),n_tokens=len(words),counter=count))
 features=[w for w,c in total.items() if len(w)>=4 and min(source_counts[s][w] for s in source_counts)>=20]
 features.sort(key=lambda w:(-total[w],w))
 if len(features)<200:raise AssertionError(f'Not enough common wordforms: {len(features)}')
 features=features[:200]
 counts=np.asarray([[p['counter'][word] for word in features] for p in pages],dtype=np.int32)
 n_tokens=np.array([p['n_tokens'] for p in pages],dtype=np.int32)
 assert all(n_tokens>0)
 for p in pages:del p['counter']
 with open(output/'pages.csv','w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=list(pages[0]));writer.writeheader();writer.writerows(pages)
 with open(output/'features.csv','w',newline='',encoding='utf-8') as f:
  w=csv.writer(f);w.writerow(['feature','overall_count','imvo_count','isig_count']);
  w.writerows((v,total[v],source_counts['IMVO'][v],source_counts['ISIG'][v]) for v in features)
 np.savez_compressed(output/'counts.npz',word_counts=counts,word_totals=n_tokens)
 per_src={s:{'issues':len(set(p['issue'] for p in pages if p['source']==s)),
             'pages':sum(p['source']==s for p in pages),
             'words':sum(p['n_tokens'] for p in pages if p['source']==s)} for s in ('IMVO','ISIG')}
 manifest={'source_zip_sha256':checksum(zip_path),'source_zip_filename':Path(zip_path).name,
           'code_schema':'P1-XH1885-issue-page-v1','tokenizer':r'casefold + re.findall(r"[^\W\d_]+", text)',
           'all_pages':len(pages),'feature_count':len(features),'by_source':per_src,
           'planned_scenarios':['SOURCE','PAGE'],'planned_replicates':200,'rng_seed':20261024}
 (output/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 print(json.dumps(manifest,indent=2))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--zip',required=True);a.add_argument('--out',required=True)
 x=a.parse_args();run(x.zip,x.out)
