from pathlib import Path
import re,regex as rx,json,pickle,hashlib
from collections import defaultdict,Counter
import numpy as np
from sklearn.linear_model import LogisticRegression

O=Path('/mnt/data/isiXhosa_phase18_evidence_routing');P=Path('/mnt/data/isiXhosa_phase17_deeper_questions'); Q=Path('/mnt/data/isiXhosa_phase15_source_grounded_search')
D=pickle.load(open('/mnt/data/isiXhosa_three_track_pilot/corpus_cache.pkl','rb'))
idx={d['id']:i for i,d in enumerate(D)}
q=[json.loads(x) for x in open(P/'language_probes_results.jsonl') if '5words_English_minor' in x]
assert len(q)==466
scores=np.load(P/'minority_bm25_scores.npz')
raw,en,xh=(scores[k] for k in ('raw','en','xh'))
qindex={z['id']:i for i,z in enumerate(q)}
# See original script for identical raw paragraph token standard.
W=rx.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
need=defaultdict(list)
for i,z in enumerate(q):need[tuple(z['text'].lower().split())].append(i)
golds=[set() for _ in q]
for j,d in enumerate(D):
 for para in re.split(r'\n\s*\n',d['raw']):
  words=[w.lower() for w in W.findall(para)]
  for k in range(len(words)-4):
   for qi in need.get(tuple(words[k:k+5]),()):golds[qi].add(j)
for i,z in enumerate(q):assert idx[z['id'].split('|')[0]] in golds[i]

splits=[]
for z in q:
 seed=int(hashlib.sha256((z['source']+'|'+z['issue']+'|PHASE18').encode()).hexdigest()[:12],16)/float(16**12)
 splits.append('dev' if seed<.60 else 'test')
splits=np.array(splits)
for grp in ('dev','test'):
 print('split',grp,'n',int((splits==grp).sum()),Counter(z['source'] for i,z in enumerate(q) if splits[i]==grp), 'issues',len({z['issue'] for i,z in enumerate(q) if splits[i]==grp}),flush=True)

n=len(q)
# Each candidate is a document selected by BM25; gold correctness requires positive score.
top={k:np.argmax(v,axis=1) for k,v in [('raw',raw),('en',en),('xh',xh)]}
mx={k:v[np.arange(n),top[k]] for k,v in [('raw',raw),('en',en),('xh',xh)]}
hit={k:np.array([int(int(top[k][i]) in golds[i] and mx[k][i]>0) for i in range(n)],dtype=int) for k in top}
# Phase17 existing LID predictions (each method trained using original corpus, not held-out test outcomes).
lid={}
for name in ('logreg_long','logreg_short','nb_long'):
 preds={x['id']:int(x['pred_xho']) for x in map(json.loads,open(P/('lid_predictions_'+name+'.jsonl'))) if x['kind']=='minority5'}
 assert len(preds)==len(q)
 lid[name]=np.array([bool(preds[z['id']]) for z in q])

baselines={'raw':hit['raw'],'oracle_en':hit['en']}
for name,predxh in lid.items():baselines['LID_'+name]=np.where(predxh,hit['xh'],hit['en'])
fuzzy={z['id']:z for z in map(json.loads,open(P/'minority_fuzzy_results.jsonl'))}
baselines['raw_fuzzy20']=np.array([int(fuzzy[z['id']]['any1']) for z in q])

# A simple unsupervised evidence router.
ratio=mx['en']/(mx['xh']+1.e-9)
def route(alpha):return np.where((alpha*mx['en']>=mx['xh']),hit['en'],hit['xh'])
alpha_grid=[.125,.25,.5,.75,1.,1.5,2.,3.,4.,8.,16.]
dev=splits=='dev';test=~dev
alphadev={str(alpha):float(route(alpha)[dev].mean()) for alpha in alpha_grid}
alpha=max(alpha_grid,key=lambda a:(alphadev[str(a)],-abs(np.log(a))))
# Record chosen alpha independent of test labels
print('chosen alpha=',alpha,'dev',alphadev[str(alpha)],'unsup_1_dev',alphadev['1.0'],flush=True)
selected_field=alpha*mx['en']>=mx['xh']
baselines['evidence_equal']=route(1.)
baselines['evidence_calibrated']=route(alpha)
# Raw fallback on uncertain relative field evidence, tau selected via development only.
margin=np.abs(alpha*mx['en']-mx['xh'])/(alpha*mx['en']+mx['xh']+1e-9)
tau_grid=[0.05,0.10,0.20,0.30,0.50]
candidates={t:np.where(margin<t,hit['raw'],route(alpha)) for t in tau_grid}
tau=max(tau_grid,key=lambda t:(candidates[t][dev].mean(),-t))
baselines['evidence_raw_fallback']=candidates[tau]
print('chosen tau',tau,'dev',float(candidates[tau][dev].mean()),flush=True)

# Lightweight supervised field selector conditional on features, only train on subset with differing field success.
# Disabled unless train issue set exhibits both classes; must beat threshold on dev by leave-issue-out CV to earn test usage.
# For strict no-test-tuning, we do not fit it for present primary result; report as not evaluated.

rows=[]
for i,z in enumerate(q):
 row={'qid':z['id'],'issue':z['issue'],'source':z['source'],'split':splits[i], 'year':int(z['issue'].split('_')[1][:4]),'page':z['id'].split('|')[0], 'front_page': int('_p001.' in z['id']), 'num_gold':len(golds[i]),'unique':len(golds[i])==1,'test_outside_lid_training':bool(z['out_training_issue']),'field_ratio':float(ratio[i]),'confidence':float(margin[i]),'selected_english':bool(selected_field[i]),'lid_logreg_short_pred_xho':bool(lid['logreg_short'][i]),'en_score_max':float(mx['en'][i]),'xh_score_max':float(mx['xh'][i]),'raw_score_max':float(mx['raw'][i])}
 for name,a in baselines.items():row[name]=int(a[i])
 rows.append(row)
with open(O/'issue_split_results.jsonl','w') as f:
 for row in rows:f.write(json.dumps(row)+'\n')

# Summaries and paired stratified issue bootstrap within each newspaper.
summary={}
for cohort,ind in [('dev',dev),('test',test)]:
 summary[cohort]={}
 for group,mask in [('all',ind),('IMVO',np.array([z['source']=='IMVO' for z in q])&ind),('ISIG',np.array([z['source']=='ISIG' for z in q])&ind),('unique',np.array([len(g)==1 for g in golds])&ind),('unique_nonfront',np.array([len(g)==1 and '_p001.' not in z['id'] for z,g in zip(q,golds)])&ind),('multi',np.array([len(g)>1 for g in golds])&ind)]:
  if mask.sum():summary[cohort][group]={'n':int(mask.sum()),'issues':len({q[i]['issue'] for i in np.where(mask)[0]}),'methods':{name:float(v[mask].mean()) for name,v in baselines.items()}}

rng=np.random.default_rng(20261023)
B=5000
comparisons=[('evidence_calibrated','raw'),('evidence_calibrated','LID_logreg_short'),('evidence_calibrated','raw_fuzzy20'),('evidence_raw_fallback','raw'),('evidence_raw_fallback','raw_fuzzy20'),('LID_logreg_short','raw')]
boots={}
for cohort,mask0 in [('test',test),('test_unique',test & np.array([len(g)==1 for g in golds])),('test_nonfront',test&np.array(['_p001.' not in z['id'] for z in q]))]:
 for group in ('IMVO','ISIG','macro_source'):
  m=mask0 if group=='macro_source' else mask0&np.array([z['source']==group for z in q])
  valid=np.where(m)[0]
  ids_by_issue={s:defaultdict(list) for s in ('IMVO','ISIG')}
  for ii in valid:ids_by_issue[q[ii]['source']][q[ii]['issue']].append(ii)
  clusters={s:list(ids_by_issue[s].values()) for s in ('IMVO','ISIG')}
  if not len(valid):continue
  weights={s:len(np.concatenate(clusters[s])) if clusters[s] else 0 for s in clusters}
  for a,b in comparisons:
   values=(baselines[a]-baselines[b]).astype(float)
   if group=='macro_source':
    sources=[s for s in ('IMVO','ISIG') if clusters[s]]
    point=float(np.mean([values[np.concatenate(clusters[s])].mean() for s in sources]))
   else:point=float(values[valid].mean())
   dist=np.empty(B,dtype=float)
   for k in range(B):
    diffs=[]
    for s in ('IMVO','ISIG'):
     if not clusters[s]:continue
     choices=rng.integers(0,len(clusters[s]),size=len(clusters[s]))
     ids=np.concatenate([clusters[s][j] for j in choices])
     diffs.append((s,float(values[ids].mean()),len(ids)))
    if group=='macro_source':dist[k]=np.mean([d[1] for d in diffs])
    else:dist[k]=np.sum([v*n for s,v,n in diffs])/np.sum([n for s,v,n in diffs])
   boots[cohort+'_'+group+'_'+a+'_minus_'+b]={'n':len(valid),'point':point,'ci95':list(map(float,np.quantile(dist,[.025,.975])))}

out={'n_query':n,'n_unique':sum(len(g)==1 for g in golds),'n_issue':len({z['issue'] for z in q}),'protocol_sha256':open(O/'precommit.sha256').read().split()[0],'alpha_grid':alpha_grid,'alpha_development_results':alphadev,'alpha_selected':alpha,'tau_selected':tau,'tau_development_results':{str(t):float(v[dev].mean()) for t,v in candidates.items()},'unsupervised_field_chosen_en_rate':float((mx['en']>=mx['xh']).mean()),'calibrated_field_chosen_en_rate':float(selected_field.mean()),'summary':summary,'bootstrap':boots,'source_labels_are_archive_fields_not_human_gold':True}
(O/'RESULTS.json').write_text(json.dumps(out,indent=2))
print('FINAL TEST summary')
for g,r in summary['test'].items():
 print(g,r['n'], {name:round(r['methods'][name]*100,1) for name in ['raw','LID_logreg_short','evidence_equal','evidence_calibrated','evidence_raw_fallback','raw_fuzzy20','oracle_en']},flush=True)
print('PAIR BOOTS')
for k,v in boots.items():
 if k.startswith('test_macro') or k.startswith('test_ISIG'):print(k,v)
