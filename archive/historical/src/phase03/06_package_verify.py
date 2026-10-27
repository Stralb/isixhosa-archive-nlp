#!/usr/bin/env python3
"""Build portable cumulative ZIP and frozen SHA256 manifest, without raw archive text."""
from pathlib import Path
import hashlib,json,zipfile,argparse

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()

def main(root,destination):
 root=Path(root);destination=Path(destination)
 manifest=root/'FILE_MANIFEST_PHASE3_SHA256.json'
 ignored={'__pycache__','.DS_Store'}
 allfiles=sorted((p for p in root.rglob('*') if p.is_file() and not any(x in ignored for x in p.parts) and p!=manifest),key=lambda x:str(x.relative_to(root)))
 hashes={str(p.relative_to(root)):sha(p) for p in allfiles}
 obj={'schema':'PRHI_PHASE3_PORTABLE_SHA256_V1','tracked_file_count':len(hashes),
      'source_data_separate':'TxtLangFiles1870-1894_TB.zip','file_hashes':hashes}
 manifest.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
 allfiles.append(manifest)
 with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in allfiles:
   name=str(Path(root.name)/p.relative_to(root))
   # fixed ZIP metadata to avoid packaging changing hashes across machines
   info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
   info.external_attr=(0o644<<16)
   with open(p,'rb') as f:z.writestr(info,f.read(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=6)
 with zipfile.ZipFile(destination) as z:
  assert z.testzip() is None
  assert len(z.namelist())==len(hashes)+1
 print('PACKAGED',str(destination),'size',destination.stat().st_size,'bytes; tracked',len(hashes),'root hash',sha(manifest))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--destination',required=True);a=p.parse_args();main(a.root,a.destination)
