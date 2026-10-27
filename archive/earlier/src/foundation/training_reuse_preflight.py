import collections, re, zipfile, json, sys
from pathlib import Path
ROOT=Path('/mnt/data/isiXhosa_publication_continuation'); sys.path.insert(0,str(ROOT))
from audit_overlap import load_pages, index_ngrams, digest
pages=load_pages(); tr=[p for p in pages if p['split']=='train']; ix,examples=index_ngrams(tr,12)
slot=dict((p['name'],int(re.search(r'_p(\d+)\.txt$',p['name']).group(1))) for p in tr)
meta=collections.defaultdict(set)
for p in tr:
    for i in range(max(0,len(p['tokens'])-11)):
        h=digest(p['tokens'][i:i+12]);meta[h].add((p['issue'],p['source'],slot[p['name']]))
counts=collections.Counter(); exemplars={}; sources={}
for p in tr:
    words=p['tokens']; pt=slot[p['name']]; src=p['source']
    masks={k:bytearray(len(words)) for k in ('allrepeat','persistent','stable','local')}
    for i in range(max(0,len(words)-11)):
        h=digest(words[i:i+12]); m=meta[h]; issue_count=len(set(row[0] for row in m))
        if issue_count<2:continue
        typ='persistent' if issue_count>=3 and len(set(row[0].split('_')[1][:7] for row in m))>=2 and sum(row[2]==pt for row in m)>=3 else 'local'
        masks['allrepeat'][i:i+12]=b'\x01'*12
        masks[typ][i:i+12]=b'\x01'*12
        if typ=='persistent' and len(exemplars)<30:
            exemplars.setdefault(h.hex(),{'text':' '.join(words[i:i+12]),'frequency':issue_count,'source':src,'page':pt})
    for k,m in masks.items(): counts[(src,k,'word')]+=sum(m)
    counts[(src,'all','word')]+=len(words)
for h,m in meta.items():
    if len(m)>=2:
        s=next(iter(m))[1]
        counts[(s,'repeat_shingles','count')]+=1
        if len(m)>=3:counts[(s,'ge3_shingles','count')]+=1
        if len(set(r[0].split('_')[1][:7] for r in m))>=2:counts[(s,'multimonth_shingles','count')]+=1
# years filenames only
cnt=collections.Counter()
with zipfile.ZipFile('/mnt/data/TxtLangFiles1870-1894_TB.zip') as z:
    for n in z.namelist():
        if '/XhTxt/' in n:
            f=n.rsplit('/',1)[-1];ma=re.match(r'^(IMVO|ISIG)_(18\d\d)-\d\d-\d\d_p\d+\.txt$',f)
            if ma:cnt[(ma.group(1),ma.group(2))]+=1
result={'train_pages_by_source':dict(collections.Counter(p['source'] for p in tr)),'train_issues_by_source':{s:len(set(p['issue'] for p in tr if p['source']==s)) for s in ('IMVO','ISIG')},'coverage':{s:{k:counts[(s,k,'word')]/counts[(s,'all','word')] for k in ('allrepeat','persistent','local')} for s in ('IMVO','ISIG')},'shingles':{s:{k:counts[(s,k,'count')] for k in ('repeat_shingles','ge3_shingles','multimonth_shingles')} for s in ('IMVO','ISIG')},'future_filename_counts':{f'{s}_{y}':n for (s,y),n in sorted(cnt.items()) if int(y)>=1891},'examples':list(exemplars.values())[:20]}
(ROOT/'training_reuse_preflight.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2)[:5000])
