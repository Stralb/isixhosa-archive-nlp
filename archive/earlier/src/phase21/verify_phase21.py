"""Independent integrity checks of Phase21 frozen files, rankings, and results.
No retraining, reselection or optimization. Recomputes correctness from saved ranks.
"""
from __future__ import annotations
import collections, hashlib, json, re, faulthandler, signal
faulthandler.register(signal.SIGUSR1)
from pathlib import Path
import numpy as np
import regex as rx
from rapidfuzz.fuzz import partial_ratio
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'phase21'
D=[json.loads(x) for x in (ROOT/'pages.jsonl').open()]
Q=[json.loads(x) for x in (OUT/'queries_frozen.jsonl').open()]
A=[json.loads(x) for x in (OUT/'anchor_audit.jsonl').open()]
P=[json.loads(x) for x in (OUT/'per_query.jsonl').open()]
RES=json.loads((OUT/'phase21_results.json').read_text())
S=json.loads((OUT/'construction_summary.json').read_text())
print("VERIFIER: loaded data",flush=True)
assert len(D)==3608 and len(Q)==len(P)==300 and len(A)==60
assert hashlib.sha256((OUT/'PRECOMMIT.md').read_bytes()).hexdigest()==RES['protocol_sha256']
assert hashlib.sha256((OUT/'queries_frozen.jsonl').read_bytes()).hexdigest()==RES['query_sha256']==S['query_file_sha256']
assert hashlib.sha256((OUT/'anchor_audit.jsonl').read_bytes()).hexdigest()==S['audit_file_sha256']
used=set()
for fn in ('isiXhosa_phase18_evidence_routing/balanced_query_results.jsonl','phase19/queries_precommit.jsonl','phase20/queries_frozen.jsonl'):
 used|={json.loads(x)['issue'] for x in (ROOT/fn).open()}
seen={q['issue'] for q in Q};assert len(seen)==60 and not seen&used
assert collections.Counter(a['source'] for a in A)=={'IMVO':30,'ISIG':30}
assert all(sum(q['anchor_id']==a['anchor_id'] for q in Q)==5 for a in A)
W=rx.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
def words(t):return [v.lower() for v in W.findall(t)]
def contains(p,toks):return any(p[i:i+len(toks)]==toks for i in range(len(p)-len(toks)+1))
idx={d['id']:i for i,d in enumerate(D)}
qby=collections.defaultdict(dict)
for q in Q:qby[q['anchor_id']][q['form']]=q
assert all(set(z)=={'keyword','clean','edit1','edit2_boundary','edit3_boundary'} for z in qby.values())
# Independently reconstruct all 60 gold sets in ONE streaming corpus pass.
first=collections.defaultdict(dict)
for audit in A:
 span=tuple(words(audit['anchor']))
 assert len(span)==13
 first[span[0]].setdefault(span,[]).append(audit['anchor_id'])
independent_gold={a['anchor_id']:set() for a in A}
for page in D:
 for paragraph in re.split(r'\n\s*\n',page['raw']):
  pt=words(paragraph)
  for j,w in enumerate(pt):
   if w not in first or j+13>len(pt):continue
   for a in first[w].get(tuple(pt[j:j+13]),[]):independent_gold[a].add(page['id'])
print("VERIFIER: single-pass gold reconstructed",flush=True)
for audit in A:
 a=audit['anchor_id'];z=qby[a];src=D[idx[audit['source_page']]]
 gold=independent_gold[a]
 assert gold and audit['source_page'] in gold and sorted(gold)==z['clean']['gold_pages']
 assert contains(words(src['raw']),words(audit['anchor']))
 assert any(contains(words(para),words(audit['anchor'])) for para in re.split(r'\n\s*\n',src['raw']))
 kw=words(z['keyword']['query'])
 assert len(kw)==6
 assert not any(contains(words(para),kw[j:j+3]) for para in re.split(r'\n\s*\n',src['xh']) for j in range(4))
 c=z['clean']['query'];assert c==audit['clean']
 for f,k in (('edit1',1),('edit2_boundary',2),('edit3_boundary',3)):
  ops=audit['edit_ops'][f];toks=c.split();assert len(ops['mutations'])==k
  assert len({op['word_pos'] for op in ops['mutations']})==k
  for op in ops['mutations']:
   x=toks[op['word_pos']]; assert x[op['char_pos']]==op['before'] and op['after']!=op['before']
   toks[op['word_pos']]=x[:op['char_pos']]+op['after']+x[op['char_pos']+1:]
  b=ops['boundary']
  if f=='edit1':assert b is None
  else:
   assert b is not None
   j=b['position']
   if b['type']=='join':toks[j:j+2]=[toks[j]+toks[j+1]]
   else:
    w=toks[j];i=b['char_pos'];toks[j:j+1]=[w[:i],w[i:]]
  assert ' '.join(toks)==z[f]['query']

print("VERIFIER: query/gold checks passed",flush=True)
scores={k:v for k,v in np.load(OUT/'phase21_scores.npz').items()}
assert all(scores[x].shape==(300,3608) for x in ('word','character','xh'))
ranks={f:np.argsort(-scores[f],axis=1,kind='stable') for f in ('word','character','xh')}
rpos={f:np.argsort(ranks[f],axis=1,kind='stable') for f in ('word','character')}
page_norm=[re.sub('[^a-z0-9]','',d['raw'].lower()) for d in D]
for i,(q,r) in enumerate(zip(Q,P)):
 if i%50==0:print("VERIFIER: score checks",i,flush=True)
 assert q['qid']==r['qid'] and q['anchor_id']==r['anchor_id'] and q['form']==r['form']
 gold=set(q['gold_pages']);origin=q['source_page'];qn=re.sub('[^a-z0-9]','',q['query'].lower());fcache={}
 def candidates(f):return [int(j) for j in ranks[f][i] if scores[f][i,j]>0]
 word=candidates('word');char=candidates('character');xh=candidates('xh')
 h=word[:15];ss=set(h)
 for j in char:
  if len(h)>=20:break
  if j not in ss:h.append(j);ss.add(j)
 for j in word:
  if len(h)>=20:break
  if j not in ss:h.append(j);ss.add(j)
 expected={'word20':word[:20],'word20_fuzzy':word[:20],
  'char20':char[:20],'char20_fuzzy':char[:20],
  'hybrid15w5c':h[:20],'hybrid15w5c_fuzzy':h[:20],
  'xh20_fuzzy':xh[:20]}
 for name,pool in expected.items():
  z=r[name]
  assert z['candidate_pool']==[D[j]['id'] for j in pool]
  assert z['candidate_count']==len(pool)<=20
  if name.endswith('_fuzzy'):
   for j in pool:
    if j not in fcache:fcache[j]=partial_ratio(qn,page_norm[j])
   ordered=sorted(pool,key=lambda j:(-fcache[j],-float(scores['word'][i,j]),j))
  elif name=='hybrid15w5c':
   ordered=sorted(pool,key=lambda j:(-1/(61+int(rpos['word'][i,j]))-1/(61+int(rpos['character'][i,j])),
                                     -float(scores['word'][i,j]),j))
  else:ordered=pool
  assert z['top1_page']==(D[ordered[0]]['id'] if ordered else None)
  assert z['recall20']==int(any(D[j]['id'] in gold for j in pool))
  assert z['top1']==int(bool(ordered) and D[ordered[0]]['id'] in gold)
  assert z['origin_top1']==int(bool(ordered) and D[ordered[0]]['id']==origin)
  assert z['top5']==int(any(D[j]['id'] in gold for j in ordered[:5]))
  assert z['first_gold_rank']==next((k+1 for k,j in enumerate(ordered) if D[j]['id'] in gold),0)

# Independently recompute reported grouped main metrics/contrasts.
for form in RES['forms']:
 for method in RES['methods']:
  for metric in ('recall20','top1','top5','origin_top1'):
   z=RES['primary_metrics'][form][method][metric]
   for source in ('IMVO','ISIG'):
    v=np.mean([r[method][metric] for r in P if r['form']==form and r['source']==source])
    assert abs(v-z['by_source'][source])<1e-5,(form,method,metric,source)
   assert abs(z['macro']-np.mean(list(z['by_source'].values())))<1e-5
A1=RES['hypotheses']['H21a_edit2_char20_minus_word20_recall20']
B1=RES['hypotheses']['H21b_edit2_hybrid_fuzzy_minus_word_fuzzy_top1']
C1=RES['hypotheses']['H21c_keyword_hybrid_minus_word_top1']
assert all(not z['pass'] for z in (A1,B1,C1))
for src in ('IMVO','ISIG'):
 sub=[r for r in P if r['source']==src and r['form']=='edit2_boundary']
 for z,a,b,key in [(A1,'char20','word20','recall20'),(B1,'hybrid15w5c_fuzzy','word20_fuzzy','top1')]:
  assert abs(z['by_source'][src]-np.mean([r[a][key]-r[b][key] for r in sub]))<1e-5
print('PASS | hashes, all 60 excluded issues, 300 frozen paired queries, paragraph-only gold, 13-token reprints, edited-query operations, exact candidates/rankings, macro metrics and hypotheses',flush=True)
print('PASS | verified scores', {k:round(v['macro'],6) for k,v in RES['hypotheses'].items()})
