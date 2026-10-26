#!/usr/bin/env python3
"""Generate a blinded *unlabelled* historical text-reuse review queue.

This is an annotation instrument, not an automatic historical genre claim.
The underlying text is the user-supplied archive; raw source context should
be inspected with the page scans before assigning labels.
"""
import collections, hashlib, json, random, re, sys
from pathlib import Path
ROOT=Path('/mnt/data/isiXhosa_publication_continuation'); sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'phase5'))
from audit_overlap import load_pages,digest
from selective_neural import shingle_occurrences
OUT=ROOT/'phase5'
SEED=20261022
RNG=random.Random(SEED)

pages=[p for p in load_pages() if p['split']=='train']
desc=shingle_occurrences(pages)
all_pos=collections.defaultdict(list)
for p in pages:
    for i in range(max(0,len(p['tokens'])-11)):
        h=digest(p['tokens'][i:i+12]); hit=desc.get(h)
        if hit:
            all_pos[h].append((p['source'],p['name'],p['issue'],i))
index={(p['source'],p['name']):p for p in pages}
bins=collections.defaultdict(list)
for h,(count,persistent) in desc.items():
    spots=all_pos[h]
    src_ct=collections.Counter(s for s,*rest in spots)
    src=src_ct.most_common(1)[0][0]
    source_spots=[s for s in spots if s[0]==src]
    if not source_spots:continue
    # Deterministic, with no reliance on the future test data.
    best=min(source_spots,key=lambda z:(z[2],z[1],z[3]))
    p=index[(best[0],best[1])]; pos=best[-1]; tokens=p['tokens']
    excerpt=' '.join(tokens[max(0,pos-25):min(len(tokens),pos+12+25)])
    core=' '.join(tokens[pos:pos+12])
    bins[(src,'persistent' if persistent else 'transient')].append({
        'code':h.hex(),'source':src,'type':'persistent' if persistent else 'transient',
        'frequency_issues':count,'first_file':best[1], 'first_issue':best[2],
        'anchor_token_position':pos,'core':core,'excerpt':excerpt,
        'word_set':set(core.split()),'pages':list(set((src,name) for src,name,issue,i in spots))})

blind=[];keys=[];stats={}
for (src,typ),items in sorted(bins.items()):
    # Group by shingle count to avoid only presenting extremely common formulae.
    items=sorted(items,key=lambda x:(x['frequency_issues'],x['code']))
    splits=[x for x in items if x['frequency_issues']>=5]
    low=[x for x in items if x['frequency_issues']<5]
    RNG.shuffle(splits);RNG.shuffle(low)
    selected=[]
    for pool,quota in ((splits,12),(low,12)):
        for item in pool:
            if len(selected)>=quota+(12 if pool is low else 0):break
            # Avoid same source-page very near positions or near-identical shingles.
            reject=False
            for old in selected:
                a,b=item['word_set'],old['word_set']
                jacc=len(a&b)/max(1,len(a|b))
                if jacc>=.65 or (item['first_file']==old['first_file'] and abs(item['anchor_token_position']-old['anchor_token_position'])<90):
                    reject=True;break
            if not reject:selected.append(item)
    if len(selected)<24:
        others=[x for x in items if x not in selected]
        RNG.shuffle(others)
        for item in others:
            if len(selected)>=24:break
            a=item['word_set']
            if all(len(a&b['word_set'])/max(1,len(a|b['word_set']))<.65 for b in selected):selected.append(item)
    # Do not invent categories or overfill when distinct candidates insufficient.
    stats[f'{src}_{typ}']={'available_shingles':len(items),'selected':len(selected)}
    for item in selected:
        code=item['code'][:10]
        # Hide heuristic category and issue frequency from experts, to avoid anchoring.
        blind.append({'annotation_id':f'REUSE_{src}_{code}',
                      'source':src,'page_file':item['first_file'],
                      'context_text':item['excerpt'],
                      'target_text':item['core'],
                      'label':'', 'confidence':'', 'notes':'',
                      'verified_against_page_image':''})
        keys.append({'annotation_id':f'REUSE_{src}_{code}',
                     'heuristic_type':item['type'],'issue_count':item['frequency_issues'],
                     'first_issue':item['first_issue'],'first_file':item['first_file'],
                     'word_anchor':item['anchor_token_position'],
                     'pages_with_same_shingle':len(set(item['pages']))})
RNG.shuffle(blind)
for fn,rows in (('annotation_queue_blind.jsonl',blind),('annotation_heuristic_keys.jsonl',keys)):
    (OUT/fn).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
(OUT/'annotation_sampling_stats.json').write_text(json.dumps(stats,indent=2))
print(json.dumps({'n_annotation_candidates':len(blind),'strata':stats},indent=2))
