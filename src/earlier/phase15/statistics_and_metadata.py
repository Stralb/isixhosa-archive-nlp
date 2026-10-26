import json, pickle, collections, math
from pathlib import Path
import numpy as np,regex
P=Path('/mnt/data/isiXhosa_three_track_pilot');O=Path('/mnt/data/isiXhosa_phase15_source_grounded_search')
D=pickle.load(open(P/'corpus_cache.pkl','rb'));Q=[json.loads(s) for s in open(O/'paragraph_queries.jsonl')]
res=[json.loads(s) for s in open(O/'paragraph_retrieval_results.jsonl')];refs={(r['method'],r['kind'],r['qid']):r for r in res}
rg=np.random.default_rng(20261023)
# hierarchical-ish bootstrap: resample entire source+issue clusters within newspaper, retaining all relevant queries
issue_src=collections.defaultdict(list)
for q in Q:issue_src[(q['source'],q['issue'])].append(q)
G={src:[v for (src2,issue),v in issue_src.items() if src==src2] for src in ('IMVO','ISIG')}

def diff_boot(method_a,method_b,kind,metric='any1',filter_fn=lambda q:True,reps=10000):
 # Precompute issue-level sums/n counts
 S={}
 for src,g in G.items():
  sums=[];sizes=[]
  for qlist in g:
   qq=[q for q in qlist if filter_fn(q)]
   if not qq:continue
   sums.append(sum(refs[(method_a,kind,q['id'])][metric]-refs[(method_b,kind,q['id'])][metric] for q in qq))
   sizes.append(len(qq))
  S[src]=(np.array(sums,dtype=float),np.array(sizes,dtype=float))
 num=sum(v[0].sum() for v in S.values());den=sum(v[1].sum() for v in S.values());observed=num/den if den else 0
 sims=np.zeros(reps);ds=np.zeros(reps)
 for src,(s,n) in S.items():
  ids=rg.integers(0,len(s),size=(reps,len(s)))
  sims+=s[ids].sum(axis=1);ds+=n[ids].sum(axis=1)
 vals=sims/ds
 return {'n':int(den),'issue_groups':int(sum(len(x[0]) for x in S.values())),
         'difference_pp':round(100*observed,3),'CI95_pp':[round(v,3) for v in 100*np.quantile(vals,[.025,.975])],
         'bootstrap_fraction_le_zero':round(float((vals<=0).mean()),4)}
RESULTS={}
for name,a,b,kind,metric,fn in [
 ('fielded_auto_minus_raw_noisy_any_top1','fielded_auto','raw','noisy','any1',lambda q:True),
 ('fielded_auto_minus_raw_noisy_exact_top1','fielded_auto','raw','noisy','exact1',lambda q:True),
 ('fielded_oracle_minus_raw_noisy_any_top1','fielded_oracle','raw','noisy','any1',lambda q:True),
 ('fielded_auto_minus_raw_clean_any_top1','fielded_auto','raw','clean','any1',lambda q:True),
 ('fielded_auto_minus_raw_noisy_nonfront_unique','fielded_auto','raw','noisy','any1',lambda q:q['source_page_num']>1 and q['num_exact_match_pages']==1),
 ('fielded_auto_minus_raw_noisy_imvo_english','fielded_auto','raw','noisy','any1',lambda q:q['source']=='IMVO' and q['lang']=='en'),
 ('fielded_auto_minus_raw_noisy_imvo_xhosa','fielded_auto','raw','noisy','any1',lambda q:q['source']=='IMVO' and q['lang']=='xh'),
]:RESULTS[name]=diff_boot(a,b,kind,metric,fn)
# Archive-wide access-policy mass, not query conditioned
W=regex.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
TOT={}
for src in ('IMVO','ISIG'):
 arr=[d for d in D if d['source']==src]
 by_issue=collections.defaultdict(lambda:collections.Counter())
 for d in arr:
  english=len(W.findall(d['en'])); xh=len(W.findall(d['xh']))
  c=by_issue[d['issue']];c['total_english']+=english;c['total_xh']+=xh
  c['english_majority_english']+=english if d['en_pct']>=50 else 0
  c['english_nonexclusive_ge10']+=english if int(d['tracker_en_words'] or 0)>=10 else 0
  c['english_nonexclusive_ge1']+=english if int(d['tracker_en_words'] or 0)>=1 else 0
  c['english_en_pct_ge20']+=english if d['en_pct']>=20 else 0
  c['english_pages']+=bool(english>0);c['english_majority_pages']+=bool(d['en_pct']>=50);c['pages']+=1
 cs=list(by_issue.values());pool=collections.Counter();
 for c in cs:pool.update(c)
 denominator=pool['total_english']
 stat={'pages':pool['pages'],'issues':len(cs),'en_pages':pool['english_pages'],'en_majority_pages':pool['english_majority_pages'],
     'english_words':pool['total_english'],'xhosa_words':pool['total_xh']}
 for key in ('english_majority_english','english_nonexclusive_ge1','english_nonexclusive_ge10','english_en_pct_ge20'):
  stat[key+'_retained_frac']=round(pool[key]/denominator,6)
 # bootstrap issues fraction hidden by Eng>=50 hypothetical policy
 keys=np.array([[c['english_majority_english'],c['total_english']] for c in cs])
 draw=rg.integers(0,len(cs),size=(10000,len(cs))); sampled=keys[draw].sum(axis=1); frac=sampled[:,0]/sampled[:,1]
 stat['english_majority_word_mass_retained_CI95']=[round(float(x),6) for x in np.quantile(frac,[.025,.975])]
 TOT[src]=stat
(O/'issue_grouped_uncertainty.json').write_text(json.dumps(RESULTS,indent=2));(O/'archive_language_access_mass.json').write_text(json.dumps(TOT,indent=2))
print('STAT RESULTS',json.dumps(RESULTS,indent=2));print('METADATA',json.dumps(TOT,indent=2))
