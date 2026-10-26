"""Post-result exploratory sampling-frame diagnostics; NOT confirmatory."""
from __future__ import annotations
import json,re,collections
from pathlib import Path
import numpy as np
import regex
R=Path(__file__).resolve().parents[1];O=R/'phase24'
W=regex.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
D={json.loads(x)['id']:json.loads(x) for x in (R/'pages.jsonl').open()}
Q=[json.loads(x) for x in (O/'queries_gold_frozen.jsonl').open()]
for q in Q:
 text=D[q['page_id']]['raw'];para=re.split(r'\n\s*\n',text)[q['paragraph_index']]
 q['paragraph_token_length']=len(W.findall(para))
 q['origin_document_words']=len(W.findall(text))
byframe={}
for frame in ('A','B'):
 rs=[q for q in Q if q['frame']==frame]
 byframe[frame]={
 'n':len(rs),'cross_issue_repeated':sum(q['cross_issue_repeat'] for q in rs),
 'median_paragraph_length_all':float(np.median([q['paragraph_token_length'] for q in rs])),
 'median_paragraph_length_repeated':float(np.median([q['paragraph_token_length'] for q in rs if q['cross_issue_repeat']])),
 'median_paragraph_length_not_repeated':float(np.median([q['paragraph_token_length'] for q in rs if not q['cross_issue_repeat']])),
 'n_paragraph_le100':sum(q['paragraph_token_length']<=100 for q in rs),
 'repeat_by_paragraph_tokens':{},
 'repeat_page_multiplicity_quantiles':np.quantile([q['exact_gold_page_count'] for q in rs if q['cross_issue_repeat']],[0,.5,.9,1]).tolist(),
 'source_count':dict(collections.Counter(q['source'] for q in rs)),
 }
 for low,high in [(50,74),(75,99),(100,149),(150,249),(250,350)]:
  sub=[q for q in rs if low<=q['paragraph_token_length']<=high]
  byframe[frame]['repeat_by_paragraph_tokens'][f'{low}-{high}']={'n':len(sub),'repeated':sum(q['cross_issue_repeat'] for q in sub),'rate':sum(q['cross_issue_repeat'] for q in sub)/len(sub) if sub else None}
# paired issue bootstrap for A minus B at fixed source stratification; descriptive not preregistered hypothesis.
issues={s:sorted({q['issue'] for q in Q if q['source']==s}) for s in ('IMVO','ISIG')};lk={(q['issue'],q['frame']):q for q in Q}
rng=np.random.default_rng(2026102424)
vals=[];boot=[]
for source in ('IMVO','ISIG'):
 delta=np.array([lk[i,'A']['cross_issue_repeat']-lk[i,'B']['cross_issue_repeat'] for i in issues[source]],float)
 vals.append(float(delta.mean()))
 idx=rng.integers(0,len(delta),size=(10000,len(delta)))
 boot.append(delta[idx].mean(axis=1))
frame_difference={'macro':float(np.mean(vals)),'by_source':dict(zip(('IMVO','ISIG'),vals)),
 'descriptive_pair_bootstrap_ci95':np.quantile(.5*(boot[0]+boot[1]),[.025,.975]).tolist(),
 'analysis':'post_result_exploratory'}
res={'posthoc_exploratory':True,'by_frame':byframe,'frame_a_minus_b_cross_issue_rate':frame_difference,
 'limitations':'No genre annotation, natural user-query distribution, or independent relevance labels. Structural units only.'}
(O/'posthoc_sampling_diagnostics.json').write_text(json.dumps(res,ensure_ascii=False,indent=2))
print(json.dumps(res,indent=2))
