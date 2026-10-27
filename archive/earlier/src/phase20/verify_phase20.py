"""Independent artifact integrity and output cross-check; does not modify results."""
import json, collections, hashlib, re
from pathlib import Path
import numpy as np
import regex as rx
P=Path(__file__).resolve().parent; R=P.parent
D=[json.loads(x) for x in (R/'pages.jsonl').open()]
Q=[json.loads(x) for x in (P/'queries_frozen.jsonl').open()]
A=[json.loads(x) for x in (P/'anchor_audit_frozen.jsonl').open()]
Z=[json.loads(x) for x in (P/'per_query_results.jsonl').open()]
F=json.loads((P/'phase20_results.json').read_text())
old={x['issue'] for f in ('isiXhosa_phase18_evidence_routing/balanced_query_results.jsonl','phase19/queries_precommit.jsonl') for x in map(json.loads,(R/f).open())}
assert len(D)==3608 and len(Q)==len(Z)==462 and len(A)==154
assert not {x['issue'] for x in Q}&old
assert len({x['issue'] for x in Q})==154
assert hashlib.sha256((P/'queries_frozen.jsonl').read_bytes()).hexdigest()==F['query_sha256']
assert hashlib.sha256((P/'PRECOMMIT.md').read_bytes()).hexdigest()==F['protocol_sha256']
assert hashlib.sha256((P/'FEASIBILITY_AMENDMENT.md').read_bytes()).hexdigest()==F['amendment_sha256']
assert F['decision']=={'H20a_pass':True,'H20b_pass':False,'no_methods_novelty_inference':True}
W=rx.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
def words(s):return [t.lower() for t in W.findall(s)]
def in_seq(seq,search):
 return any(seq[k:k+len(search)]==search for k in range(len(seq)-len(search)+1))
idx={d['id']:d for d in D};aud={a['anchor_id']:a for a in A}
for r in Q:
 assert r['source_page'] in idx
 a=aud[r['anchor_id']];view=r['view']
 assert a['issue']==r['issue'] and a['source']==r['source']
 paragraph=re.split(r'\n\s*\n',idx[r['source_page']][view])[a['source_para_index']]
 pt=words(paragraph);span=r['anchor_evidence_tokens'].split()
 assert len(span)==13 and in_seq(pt,span)
 assert any(in_seq(words(p),span) for p in re.split(r'\n\s*\n',idx[r['source_page']]['raw']))
 if r['form']=='clean':assert r['query']==' '.join(span[4:9])
 elif r['form']=='keyword':
  qt=words(r['query']);assert len(qt)==6 and all(t in pt for t in qt)
  assert all(not in_seq(pt,qt[k:k+3]) for k in range(len(qt)-2))
 elif r['form']=='synthetic_noise':
  assert r['query']!=' '.join(span[4:9])
  assert len(r['query'].split()) in (4,5,6)
 else:raise AssertionError('unexpected form')

mat=np.load(P/'phase20_bm25_scores.npz')
assert all(mat[f].shape==(462,3608) for f in ('raw','en','xh'))
id_to_pos={d['id']:k for k,d in enumerate(D)}
for k,(q,z) in enumerate(zip(Q,Z)):
 assert q['qid']==z['qid'] and q['source_page']==z['source_page']
 ranks=np.argsort(-mat['raw'][k],kind='stable')
 first=next(j for j in ranks if mat['raw'][k,j]>0)
 assert z['raw_bm25']['top1_page']==D[int(first)]['id']
 for name in ('raw_bm25','raw_fullfuzzy','gated_fullfuzzy','unconditional_fullfuzzy','hard_fullfuzzy','raw_local120','gated_local120'):
  result=z[name];cand=z['candidate_pools'][name]
  assert len(cand)<=20
  assert result['top1_page'] in cand if result['top1_page'] else not cand
  assert result['anchor_top1']==int(result['top1_page']==q['source_page'])
  assert result['top1']==int(result['top1_page'] in z['gold_pages'])
  assert result['anchor_recall20']==int(q['source_page'] in cand)
  assert result['recall20']==int(bool(set(cand)&set(z['gold_pages'])))

# Rebuild source-view macro Top-1 independently, plus H20a/H20b point estimates.
groups=('IMVO_en','ISIG_en','IMVO_xh','ISIG_xh');forms=('keyword','clean','synthetic_noise')
for form in forms:
 for name in ('raw_bm25','raw_fullfuzzy','gated_fullfuzzy','unconditional_fullfuzzy','hard_fullfuzzy','raw_local120','gated_local120'):
  point=np.mean([np.mean([z[name]['anchor_top1'] for z in Z if z['form']==form and z['group']==g]) for g in groups])
  stored=F['method_metrics'][form+'_'+name]['anchor_top1']['macro']
  assert np.isclose(point,stored,atol=1e-7)

by_a=collections.defaultdict(dict)
for z in Z:by_a[z['anchor_id']][z['form']]=z
h20a=np.mean([np.mean([(a['synthetic_noise']['raw_fullfuzzy']['anchor_top1']-a['synthetic_noise']['raw_bm25']['anchor_top1'])-(a['keyword']['raw_fullfuzzy']['anchor_top1']-a['keyword']['raw_bm25']['anchor_top1']) for a in by_a.values() if a['keyword']['group']==g]) for g in groups])
h20b=np.mean([np.mean([a['keyword']['raw_local120']['anchor_top1']-a['keyword']['raw_bm25']['anchor_top1'] for a in by_a.values() if a['keyword']['group']==g]) for g in groups])
assert np.isclose(h20a,F['comparisons']['H20a_interaction']['macro'])
assert np.isclose(h20b,F['comparisons']['H20b_keyword_local_minus_bm25']['macro'])
print(json.dumps({'verification':'PASS','pages':3608,'source_issues':154,'queries':462,'sha256_integrity':True,'gold_sentence_boundary_integrity':True,'source_issues_new':True,'threegram_nonverbatim_audit':True,'raw_bm25_candidate_order':True,'all_method_metrics_reaggregated':True,'interaction_pp':round(100*h20a,2),'local_transfer_pp':round(100*h20b,2)},indent=2))
