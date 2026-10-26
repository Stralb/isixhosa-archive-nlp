"""Rebuild page-aligned raw/Xh/En records and precommitted query set from supplied archive files."""
from pathlib import Path
from collections import Counter,defaultdict
import zipfile, csv, io, json, hashlib, random, pickle, re, statistics
import regex as rx
ROOT=Path('/mnt/data'); OUT=ROOT/'isiXhosa_three_track_pilot'; OUT.mkdir(exist_ok=True)
raw=zipfile.ZipFile(ROOT/'1. TxtFiles1870-1894_TB.zip')
lang=zipfile.ZipFile(ROOT/'TxtLangFiles1870-1894_TB.zip')
def select(z,p):
 d={}
 for n in z.namelist():
  if '/__MACOSX/' in n or n.startswith('__MACOSX/') or not n.lower().endswith('.txt'):continue
  if p not in n:continue
  k=n.rsplit('/',1)[-1]
  if not re.match(r'^(IMVO|ISIG)_\d{4}-\d\d-\d\d_p\d\d\d.txt$',k):continue
  d[k]=z.read(n).decode('utf-8-sig','replace').replace('\r\n','\n').replace('\r','\n')
 return d
raws=select(raw,'/TxtFiles1870-1894_TB/'); xhs=select(lang,'/XhTxt/'); ens=select(lang,'/EnTxt/')
print('file counts',len(raws),len(xhs),len(ens),flush=True)
assert len(raws)==len(xhs)==len(ens)==3608
trackername=next(n for n in lang.namelist() if n.endswith('LanguageSplitTracker70-94.csv'))
table={r['Edition']:r for r in csv.DictReader(io.StringIO(lang.read(trackername).decode('utf-8-sig','replace')))}
print('tracker records',len(table),flush=True)
records=[]
for k in sorted(raws):
 stem=k[:-4]; parts=stem.split('_'); source=parts[0];date=parts[1]; pg=parts[2]
 t=table.get(k,{})
 def pct(label):
  try: return float(t.get(label,'0') or '0')
  except ValueError:return 0.
 records.append(dict(id=k,source=source,date=date,year=int(date[:4]),issue=f'{source}_{date}',page=pg,
                     raw=raws[k],xh=xhs[k],en=ens[k],xh_pct=pct('Xh%'),en_pct=pct('En%'),
                     tracker_xh_words=t.get('Xho_Wordcount'),tracker_en_words=t.get('Eng_Wordcount')))
with open(OUT/'corpus_cache.pkl','wb') as f:pickle.dump(records,f,protocol=5)
word_re=rx.compile(r'(?V1)\p{L}+(?:[\'’]\p{L}+)*')
counts=Counter(r['source'] for r in records)
langcounts={}
for s in ['IMVO','ISIG']:
 arr=[r for r in records if r['source']==s]
 langcounts[s]={
  'pages':len(arr),'issues':len(set(r['issue'] for r in arr)),
  'raw_chars':sum(len(r['raw']) for r in arr),
  'xh_word_occurrences':sum(len(word_re.findall(r['xh'])) for r in arr),
  'en_word_occurrences':sum(len(word_re.findall(r['en'])) for r in arr),
  'pages_with_50pct_both_languages':sum(r['xh_pct']>=20 and r['en_pct']>=20 for r in arr),
  'pages_with_nontrivial_both_language_views':sum(len(word_re.findall(r['xh']))>=50 and len(word_re.findall(r['en']))>=50 for r in arr),
  'english_majority_pages':sum(r['en_pct']>=50 for r in arr),
  'xhosa_majority_pages':sum(r['xh_pct']>=50 for r in arr),
 }
Q=[]
for s in ('IMVO','ISIG'):
 for label in ('xh','en'):
  cand=[]
  for r in records:
   if r['source'] != s:continue
   text=r[label]
   words=word_re.findall(text)
   if len(words)<50:continue
   rng=random.Random(int(hashlib.sha256((r['id']+label+'20261023').encode()).hexdigest()[:16],16))
   # Choose from all words; no ranking-based filtering or hand-picked top terms.
   for attempt in range(10):
    idx=rng.randrange(0,len(words)-10)
    w=[w.lower() for w in words[idx:idx+10]]
    if len(set(w))>=5 and sum(len(x) for x in w)>42:break
   else: continue
   cand.append(dict(id=f'{r["id"]}|{label}',page=r['id'],source=s,issue=r['issue'],lang=label,query=' '.join(w)))
  rng=random.Random(20261023+(17 if s=='IMVO' else 43)+(0 if label=='xh' else 11))
  rng.shuffle(cand)
  Q.extend(cand[:180])

def synthetic_ocr(query,label):
 words=query.split();rng=random.Random(int(hashlib.sha256((label+'#noise20261023').encode()).hexdigest()[:16],16))
 eligible=[i for i,w in enumerate(words) if len(w)>=4]
 rng.shuffle(eligible)
 for idx in eligible[:max(1,round(len(words)*.25))]:
  w=words[idx]; pos=rng.randrange(1,len(w)-1)
  choices='aeinorsutkgmhbd';letter=rng.choice([c for c in choices if c!=w[pos]])
  words[idx]=w[:pos]+letter+w[pos+1:]
 # one word-boundary perturbation per query, if possible
 k=rng.randrange(1,len(words)-1)
 if rng.randrange(2)==0 and len(words[k])>5:
  pos=rng.randrange(2,len(words[k])-1)
  words[k:k+1]=[words[k][:pos],words[k][pos:]]
 else:
  words[k:k+2]=[words[k]+words[k+1]]
 return ' '.join(words)
for q in Q:q['noisy_query']=synthetic_ocr(q['query'],q['id'])
with open(OUT/'queries.jsonl','w',encoding='utf-8') as f:
 for q in Q:f.write(json.dumps(q,ensure_ascii=False)+'\n')
audit=dict(num_raw_pages=len(records),num_unique_issues=len(set(r['issue'] for r in records)),counts=langcounts,
          query_counts={f'{s}_{lg}':sum(q['source']==s and q['lang']==lg for q in Q) for s in ('IMVO','ISIG') for lg in ('xh','en')},
          precommit_sha256=hashlib.sha256((OUT/'PRECOMMIT.md').read_bytes()).hexdigest())
(OUT/'dataset_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2),flush=True)
