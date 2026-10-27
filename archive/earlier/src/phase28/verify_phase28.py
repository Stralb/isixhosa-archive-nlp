#!/usr/bin/env python3
"""Independently verify Phase28 saved pair identities, score calculations and optimality.
This is NOT a new statistical hypothesis test and doesn't claim semantic validity.
"""
import csv, json, hashlib, collections, re, os
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
D=Path(__file__).parent;P=Path(os.environ.get('PHASE26_DIR',str(D.parent/'isiXhosa_phase26_culture_pilots')))

def csvrows(n):
 with open(D/n,encoding='utf8',newline='') as f:return list(csv.DictReader(f))

w=[json.loads(s) for s in (P/'windows.jsonl').read_text(encoding='utf8').splitlines()]
byfile={x['file']:x for x in w}
pairs=csvrows('all_year_matched_pairs.csv'); assert len(pairs)==249
saved=json.loads((D/'RESULTS.json').read_text(encoding='utf8'))
prehash=(D/'FROZEN_SHA256.txt').read_text(encoding='utf8').splitlines()
for line in prehash:
 digest,path=line.split(maxsplit=1)
 check=(D/'PRECOMMIT.md') if Path(path).name=='PRECOMMIT.md' else (P/Path(path).name)
 assert check.exists(),('missing frozen file',check)
 assert hashlib.sha256(check.read_bytes()).hexdigest()==digest,(str(check),'changed')
assert len([x for x in pairs if int(x['year'])==1888])==84
# Check no repeated file and same-month pairs in each year
for yy in (1885,1887,1888):
 q=[x for x in pairs if int(x['year'])==yy]
 for field in ('imvo_file','isig_file'):
  assert len({r[field] for r in q})==len(q)
 for r in q:
  assert byfile[r['imvo_file']]['source']=='IMVO' and byfile[r['isig_file']]['source']=='ISIG'
  assert byfile[r['imvo_file']]['year']==byfile[r['isig_file']]['year']==yy
  assert byfile[r['imvo_file']]['month']==byfile[r['isig_file']]['month']==int(r['month'])

# Independently compute actual pair cosine values and optimal matching per month
SUBSTR=('imvo','isigidi','samaxosa','zabantsundu','bantsundu','ntsundu','mnyama','xosa','jabavu','gqoba','makiwane')
TOK=re.compile(r'[^\W\d_]+',re.U)
def content(w):return ' '.join(t for t in TOK.findall(w['text'].lower()) if t not in {'lovedale','mr'} and not any(s in t for s in SUBSTR))
vec=TfidfVectorizer(ngram_range=(1,1),min_df=2,max_df=.35,sublinear_tf=True,norm='l2',max_features=15000,token_pattern=r'(?u)\b[^\W\d_]+\b')
vec.fit([content(x) for x in w if x['year'] in (1885,1887)])
x=vec.transform([content(i) for i in w]);ix={v['file']:k for k,v in enumerate(w)}
max_abs=0.0
for yy in (1885,1887,1888):
 for mm in range(1,13):
  a=[k for k,v in enumerate(w) if v['year']==yy and v['month']==mm and v['source']=='IMVO']
  b=[k for k,v in enumerate(w) if v['year']==yy and v['month']==mm and v['source']=='ISIG']
  sim=(x[a] @ x[b].T).toarray();row,col=linear_sum_assignment(-sim)
  sample=[p for p in pairs if int(p['year'])==yy and int(p['month'])==mm]
  assert len(sample)==min(len(a),len(b))
  obs=[]
  for p in sample:
   val=float(sim[a.index(ix[p['imvo_file']]),b.index(ix[p['isig_file']])]);max_abs=max(max_abs,abs(val-float(p['cosine'])));obs.append(val)
  assert abs(sum(obs)-sum(sim[row,col]))<1e-8,(yy,mm)
assert max_abs<1e-11

# Completely independently summarize matched page-level saved predictions/costs
rows=csvrows('matched_1888_page_scores.csv'); assert len(rows)==168
assert len({x['file'] for x in rows})==168
metrics={}
for cls in ['char','word']:
 for ss in ('IMVO','ISIG'):
  grp=collections.defaultdict(list)
  for r in rows:
   if r['source']==ss:grp[r['issue']].append(float(r[f'{cls}_source_correct']))
  metrics[(cls,ss)]=float(np.mean([np.mean(v) for v in grp.values()]))
 char=saved['classification']['char_wb_3_5']
assert abs(.5*(metrics['char','IMVO']+metrics['char','ISIG'])-char['matched']['macro'])<1e-12
assert abs(.5*(metrics['word','IMVO']+metrics['word','ISIG'])-saved['classification']['word_unigram']['matched']['macro'])<1e-12
bpe={}
for ss in ('IMVO','ISIG'):
 seedvals=[]
 for seed in (0,1,2):
  grp=collections.defaultdict(list)
  for r in rows:
   if r['source']==ss:grp[r['issue']].append(float(r[f'bpe_advantage_seed{seed}']))
  seedvals.append(float(np.mean([np.mean(v) for v in grp.values()])))
 bpe[ss]=float(np.mean(seedvals))
 assert abs(bpe[ss]-saved['bpe']['matched_savings'][ss])<1e-12

# Verify numeric hypothesis gates, positive intervals and mean similarity
assert saved['preregistered_outcomes']['H28a']==(saved['match_quality_1888']['cosine_gain_over_random']>=.025 and saved['match_quality_1888']['month_bootstrap95'][0]>0)
assert saved['preregistered_outcomes']['H28b']==(saved['classification']['char_wb_3_5']['matched']['macro']>.65 and saved['matched_classification_month_bootstrap95'][0]>.5)
assert saved['preregistered_outcomes']['H28c']==all(saved['bpe']['month_bootstrap95'][s][0]>0 for s in ('IMVO','ISIG'))
quality=csvrows('monthly_matching_quality.csv')
assert len(quality)==36
for y in (1885,1887,1888):
 assert sum(int(q['n_pairs']) for q in quality if int(q['year'])==y)==len([p for p in pairs if int(p['year'])==y])
verified={'passed':True,'prior_hashes_still_match':True,'n_pairs_all_years':len(pairs),'n_pairs_1888':84,
          'n_1888_scored_pages':len(rows),'max_pair_cosine_difference':max_abs,
          'matching_optimality_verified_for_year_month_groups':36,
          'independent_issue_balanced_char':.5*(metrics['char','IMVO']+metrics['char','ISIG']),
          'independent_issue_balanced_word':.5*(metrics['word','IMVO']+metrics['word','ISIG']),
          'independent_bpe_advantage':bpe,'preregistered_gate_logic_verified':True}
(D/'VERIFICATION.json').write_text(json.dumps(verified,indent=2))
print(json.dumps(verified,indent=2))
