from pathlib import Path
import pickle,json,regex,collections, numpy as np
O=Path('/mnt/data/isiXhosa_three_track_pilot')
pages=pickle.load(open(O/'corpus_cache.pkl','rb')); d={p['id']:p for p in pages}; q=[json.loads(s) for s in open(O/'queries.jsonl')]
pat=regex.compile(r'(?V1)\p{L}+(?:[\'’]\p{L}+)*')
valid={}
for z in q:
 p=z['page']; key=z['id']; phrase=' '.join(w.lower() for w in pat.findall(z['query']))
 rawtext=' '.join(w.lower() for w in pat.findall(d[p]['raw']))
 valid[key]=bool(phrase in rawtext)
(O/'query_exact_presence.json').write_text(json.dumps(valid,indent=2))
print('VERBATIM',sum(valid.values()),len(valid),flush=True)
res=[json.loads(s) for s in open(O/'retrieval_strong_results.jsonl')]
for method in ('BM25_full_raw','BM25_full_xh_only','Char_truncated_raw','BM25_filter_en_majority','BM25_filter_en_multitag_50words'):
 for kind in ('clean','noisy'):
  for lang in ('en','xh'):
   for truth in (True,False):
    z=[r for r in res if r['method']==method and r['kind']==kind and r['lang']==lang and valid[r['qid']]==truth]
    if not z:continue
    print(method,kind,lang,'valid',truth,'N',len(z),'R1',round(np.mean([r['r1'] for r in z]),3),'R5',round(np.mean([r['r5'] for r in z]),3))
