"""Retrospective near-duplicate text checks. NOT a frozen evaluation endpoint."""
import json,re
from pathlib import Path
from rapidfuzz import fuzz
P=Path(__file__).parent;ROOT=P.parent
R=[json.loads(l) for l in (P/'per_query.jsonl').open()]
D={json.loads(l)['id']:json.loads(l)['raw'] for l in (ROOT/'pages.jsonl').open()}
def norm(s):return re.sub('[^a-z0-9]','',s.lower())
N={k:norm(s) for k,s in D.items()}
audit=[]
for i in range(0,len(R),2):
 e,c=R[i],R[i+1];q=norm(c['text'])
 for m in ('BM25','BM25_FUZZY20','CHAR3_TFIDF','CHAR3_FUZZY20','XH_BM25_FUZZY20'):
  item=e['methods'][m];p=item['top1_page']
  sim=fuzz.partial_ratio(q,N[p]) if p else None
  match20=max((fuzz.partial_ratio(q,N[k]) for k in item['candidates']),default=0.0)
  audit.append({'page_id':e['page_id'],'mode':'external','method':m,
   'strict_origin_top1':item['top1'],'top1_page':p,'corrected_query_fuzzy_similarity_on_predicted_page':sim,
   'max_corrected_query_partial_ratio_in_top20':match20,
   'posthoc_threshold_for_similar_content':95,
   'predicted_page_has_near_identical_content_not_human_adjudicated':bool(sim is not None and sim>=95)})
(P/'posthoc_duplicate_diagnostics.json').write_text(json.dumps(audit,indent=2))
for m in ('BM25','BM25_FUZZY20','CHAR3_TFIDF','CHAR3_FUZZY20','XH_BM25_FUZZY20'):
 a=[x for x in audit if x['method']==m];bad=[x for x in a if not x['strict_origin_top1']]
 print(m,'missed strict origin',len(bad),'of which near-duplicate content',sum(x['predicted_page_has_near_identical_content_not_human_adjudicated'] for x in bad),'content match Top20 >=95',sum(x['max_corrected_query_partial_ratio_in_top20']>=95 for x in a))
