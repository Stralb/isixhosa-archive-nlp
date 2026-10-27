import re,regex as rx,hashlib,json,pickle
from pathlib import Path
from collections import Counter
P=Path('/mnt/data/isiXhosa_phase18_evidence_routing');Q=Path('/mnt/data/isiXhosa_phase17_deeper_questions')
D=pickle.load(open('/mnt/data/isiXhosa_three_track_pilot/corpus_cache.pkl','rb'));docs={d['id']:d for d in D}
E=[json.loads(x) for x in open(Q/'language_probes_results.jsonl') if '5words_English_minor' in x];assert len(E)==466
W=rx.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
def words(txt):return [w.lower() for w in W.findall(txt)]
paired=[];rejected=[]
for e in E:
 page=e['id'].split('|')[0];d=docs[page];rawp=[set(tuple(z[i:i+5]) for i in range(len(z)-4)) for z in [[*words(p)] for p in re.split(r'\n\s*\n',d['raw'])]]
 valid=set().union(*rawp)
 candidates=[]
 for pp in re.split(r'\n\s*\n',d['xh']):
  a=words(pp)
  for j in range(len(a)-4):
   seq=tuple(a[j:j+5])
   if seq in valid and len(seq)==5: candidates.append(seq)
 if not candidates:rejected.append(page);continue
 # Select stable hash ranked candidate without token scores or source page retrieval outcomes.
 candidates=list(dict.fromkeys(candidates))
 winner=min(candidates,key=lambda z:hashlib.sha256((page+'|'+ ' '.join(z)+'|PHASE18XH').encode()).hexdigest())
 paired.append({'id':page+'|xhminor','kind':'5words_Xhosa_minor','issue':e['issue'],'source':e['source'],'lang':'xh','text':' '.join(winner),'source_page':page,'paired_en_id':e['id']})
(P/'balanced_xh_queries.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in paired))
(P/'balanced_en_queries.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in E if x['id'].split('|')[0] not in rejected))
(P/'balanced_build_summary.json').write_text(json.dumps({'english_total':len(E),'matched_pairs':len(paired),'skipped_pages':rejected,'source_counts':dict(Counter(x['source'] for x in paired))},indent=2))
print('Pairs',len(paired),'skipped',rejected,'source',Counter(x['source'] for x in paired))
