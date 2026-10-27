#!/usr/bin/env python3
"""Training-only, exploratory issue-frequency matched reuse sensitivity."""
import collections, json, random, sys
from pathlib import Path
ROOT=Path('/mnt/data/isiXhosa_publication_continuation');sys.path.insert(0,str(ROOT))
from audit_overlap import load_pages,digest
PAGES=[p for p in load_pages() if p['split']=='train']

def coverage(pages):
    ix=collections.defaultdict(set)
    for p in pages:
        tok=p['tokens']
        for i in range(max(0,len(tok)-11)):
            ix[digest(tok[i:i+12])].add(p['issue'])
    words=0;repeat=0
    for p in pages:
        tok=p['tokens'];seen=bytearray(len(tok))
        for i in range(max(0,len(tok)-11)):
            if len(ix[digest(tok[i:i+12])])>=2:seen[i:i+12]=b'\x01'*12
        repeat+=sum(seen);words+=len(tok)
    return {'covered_word_fraction':repeat/words if words else 0,'tokens':words,'pages':len(pages),'issues':len(set(p['issue'] for p in pages))}

imvo=[p for p in PAGES if p['source']=='IMVO'];isig=[p for p in PAGES if p['source']=='ISIG']
isig0=coverage(isig)
months=collections.defaultdict(list)
for p in imvo:months[p['issue'].split('_')[1][:7]].append(p)
assert len(months)==9
choices={m:sorted(set(p['issue'] for p in ps)) for m,ps in months.items()}
vals=[]
for k in range(40):
    rng=random.Random(20261022+k)
    chosen={rng.choice(choices[m]) for m in sorted(months)}
    v=coverage([p for p in imvo if p['issue'] in chosen]);v['seed']=k
    vals.append(v)
r={'warning':'exploratory training-only sensitivity: 9 monthly issues each, not equal word/character budget',
   'unmatched_imvo':coverage(imvo),'unmatched_isig':isig0,
   'matched_imvo_mean_fraction':sum(x['covered_word_fraction'] for x in vals)/len(vals),
   'matched_imvo_min':min(x['covered_word_fraction'] for x in vals),
   'matched_imvo_max':max(x['covered_word_fraction'] for x in vals),
   'matched_imvo_mean_tokens':sum(x['tokens'] for x in vals)/len(vals),
   'samples':vals}
(ROOT/'phase5'/'issue_count_sensitivity.json').write_text(json.dumps(r,indent=2))
print(json.dumps({k:v for k,v in r.items() if k!='samples'},indent=2))
