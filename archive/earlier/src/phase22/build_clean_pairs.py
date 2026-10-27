"""Build metadata-aligned clean query windows without any corpus-wide retrieval scores."""
import hashlib,json
from pathlib import Path
import regex as re
from rapidfuzz import fuzz
ROOT=Path(__file__).parent.parent
OUT=Path(__file__).parent
assert hashlib.sha256((OUT/'queries_external_frozen.jsonl').read_bytes()).hexdigest()==(OUT/'QUERY_SHA256.txt').read_text().split()[0]
D={json.loads(line)['id']:json.loads(line) for line in (ROOT/'pages.jsonl').open()}
Q=[json.loads(line) for line in (OUT/'queries_external_frozen.jsonl').open()]
W=re.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")

def words(x):return W.findall(x.lower())

def make_pair(q):
 doc=D[q['page_id']]['raw'];t=words(doc);x=words(q['query_external']);n=len(x)
 best=None;best_key=None
 for k in range(max(5,n-3),n+4):
  for i in range(len(t)-k+1):
   cand=t[i:i+k]
   sim=fuzz.ratio(' '.join(x),' '.join(cand))
   score=(sim,-abs(k-n),-i)
   if best_key is None or score>best_key:
    best_key=score;best=(i,k,cand)
 i,k,cand=best
 return {**q,'query_clean':' '.join(cand),'clean_start_word':i,'clean_window_words':k,
  'alignment_ratio_0_100':float(best_key[0]),'corrected_page_has_external_excerpt_verbatim':bool(' '.join(x) in ' '.join(t)),
  'external_word_count':n,'source_page_word_count':len(t)}

paired=[make_pair(q) for q in Q]
with (OUT/'paired_queries_frozen.jsonl').open('w') as f:
 for r in paired:f.write(json.dumps(r,ensure_ascii=False)+'\n')
sha=hashlib.sha256((OUT/'paired_queries_frozen.jsonl').read_bytes()).hexdigest()
(OUT/'PAIRS_SHA256.txt').write_text(sha+'  paired_queries_frozen.jsonl\n')
print('Saved paired queries, SHA256',sha)
for r in paired:
 print('\n'+r['page_id'],'alignment',r['alignment_ratio_0_100'],'verbatim?',r['corrected_page_has_external_excerpt_verbatim'])
 print('EXTERNAL:',r['query_external']);print('CLEAN:',r['query_clean'])
