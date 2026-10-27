"""Download three SOURCE corpora with provenance; run on internet-connected PC.

Historical article DOI records: 10.25375/uct.22332277 and 10.25375/uct.22332286.
Modern CC-BY 4.0 corpus: dsfsi/vukuzenzele-nlp, xho.jsonl.
"""
import argparse, hashlib, json, os, sys
from pathlib import Path
from urllib.parse import urlparse
import requests

SOURCES=[
 {'id':'ityala_lamawele','article':22332277,'doi':'10.25375/uct.22332277','type':'historical'},
 {'id':'zemkinkomo_magwalandini','article':22332286,'doi':'10.25375/uct.22332286','type':'historical'},
 {'id':'vukuzenzele_xho','type':'modern','url':'https://raw.githubusercontent.com/dsfsi/vukuzenzele-nlp/master/data/monolingual_jsonl/xho.jsonl',
  'git_blob_sha1':'99740ae0b41fa12e0bf7cee686391cabff72ff1f','license':'CC-BY-4.0'}]

def sha256(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def download(url,path):
 path.parent.mkdir(parents=True, exist_ok=True)
 if not path.exists() or path.stat().st_size==0:
  with requests.get(url,timeout=90,stream=True,headers={'User-Agent':'ResearchCorpusCollector/1.0'}) as r:
   r.raise_for_status()
   with open(str(path)+'.partial','wb') as f:
    for chunk in r.iter_content(256*1024):
     if chunk: f.write(chunk)
  os.replace(str(path)+'.partial',path)
 return {'file':str(path),'bytes':path.stat().st_size,'sha256':sha256(path),'source_url':url}

def get_book(s,folder):
 api=f'https://api.figshare.com/v2/articles/{s["article"]}'
 r=requests.get(api,timeout=45,headers={'User-Agent':'ResearchCorpusCollector/1.0'}); r.raise_for_status();meta=r.json()
 # DOI can be absent for some institutional Figshare mirrors; log rather than assume.
 reported_doi=str(meta.get('doi') or meta.get('resource_doi') or '')
 if reported_doi and s['doi'].lower() not in reported_doi.lower():
  raise ValueError(f'DOI mismatch! Expected {s["doi"]}, received {reported_doi}')
 files=meta.get('files') or []
 if not files: raise ValueError(f'No downloadable files at {api}; open DOI landing page manually.')
 allowed={'.zip','.docx','.txt','.doc'}
 selected=[f for f in files if Path(f['name']).suffix.lower() in allowed]
 if not selected:
  raise ValueError('No text or Word files shown by Figshare API. Manual acquisition needed.')
 rec={'id':s['id'],'doi':s['doi'],'metadata_api':api,'reported_doi':reported_doi,
      'article_title':meta.get('title'),'license':meta.get('license'),'version':meta.get('version'),
      'all_file_count':len(files),'files':[]}
 for f in selected:
  name=Path(f['name']).name
  if f.get('size',0)>150_000_000: raise ValueError(f'Unexpectedly large file: {name}')
  # Prefix file ID to avoid filename collisions
  p=folder / f'{f.get("id", "unknown")}_{name}'
  rec['files'].append(download(f['download_url'],p))
 return rec

def main():
 a=argparse.ArgumentParser();a.add_argument('--out',default='external_raw');a.add_argument('--only',choices=['all','historical','modern'],default='all');args=a.parse_args()
 root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
 report=[]
 for s in SOURCES:
  if args.only!='all' and s['type']!=args.only: continue
  d=root/s['id']; d.mkdir(parents=True,exist_ok=True)
  try:
   if s['type']=='historical': rec=get_book(s,d)
   else:
    f=download(s['url'],d/'xho.jsonl')
    data=(d/'xho.jsonl').read_bytes(); digest=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\x00'+data).hexdigest()
    if digest != s['git_blob_sha1']: raise ValueError(f'Git blob changed ({digest}); pin an audited revision before analysis')
    rec={'id':s['id'],'git_blob_sha1':digest,'license':'CC BY 4.0','files':[f]}
   rec['status']='downloaded'; print('OK',s['id'],len(rec['files']),'files')
  except Exception as e:
   rec={'id':s['id'],'status':'failed','error':repr(e),'source':s,'manual_url':('https://doi.org/'+s['doi']) if s.get('doi') else s['url']}
   print('FAILED',s['id'],rec['error'],file=sys.stderr)
  report.append(rec)
 (root/'download_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 print('Saved',root/'download_manifest.json')
 if any(x['status']=='failed' for x in report):return 1
 return 0
if __name__=='__main__':sys.exit(main())
