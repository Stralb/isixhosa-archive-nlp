from pathlib import Path
import pickle,json,regex as rx,re,numpy as np
from collections import defaultdict,Counter
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as sp
P=Path('/mnt/data/isiXhosa_phase17_deeper_questions')
D=pickle.load(open('/mnt/data/isiXhosa_three_track_pilot/corpus_cache.pkl','rb'))
q=[json.loads(x) for x in open(P/'language_probes_results.jsonl') if '5words_English_minor' in x]
M=len(q);N=len(D);assert M==466
idx={d['id']:i for i,d in enumerate(D)}
W=rx.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
patterns=defaultdict(list)
for i,x in enumerate(q):patterns[tuple(x['text'].lower().split())].append(i)
gold=[set() for _ in q]
for j,d in enumerate(D):
 for para in re.split(r'\n\s*\n',d['raw']):
  toks=[w.lower() for w in W.findall(para)]
  for k in range(len(toks)-4):
   zs=patterns.get(tuple(toks[k:k+5]))
   if zs:
    for z in zs:gold[z].add(j)
for i,x in enumerate(q):
 assert idx[x['id'].split('|')[0]] in gold[i]
print('Multiple golds',sum(len(v)>1 for v in gold),'unique',sum(len(v)==1 for v in gold),flush=True)
# Phase 15 BM25 exact matching same setup. No new tuning.
score={}
for field in ('raw','en','xh'):
 print('Fit',field,flush=True)
 v=CountVectorizer(lowercase=True,token_pattern=r'(?u)\b[^\W\d_]+\b',dtype=np.float32)
 X=v.fit_transform([d[field] for d in D]).tocsr()
 lengths=np.asarray(X.sum(axis=1)).ravel().astype(np.float32);df=np.asarray((X>0).sum(axis=0)).ravel();idf=np.log1p((N-df+.5)/(df+.5));mu=max(float(lengths.mean()),1)
 c=X.tocoo();wt=c.data*2.2/(c.data+1.2*(.25+.75*lengths[c.row]/mu))*idf[c.col]
 weights=sp.csr_matrix((wt,(c.row,c.col)),shape=c.shape,dtype=np.float32)
 T=v.transform([x['text'] for x in q]);T.data[:]=1
 score[field]=(T@weights.T).toarray().astype(np.float32)
 del v,X,c,T,weights
np.savez_compressed(P/'minority_bm25_scores.npz',raw=score['raw'],en=score['en'],xh=score['xh'])
score['oracle_en']=score['en']
score['auto']=np.where(np.array([x['pred_xho'] for x in q])[:,None],score['xh'],score['en'])
res=[];summ={}
for name,s in score.items():
 rows=[]
 for i,x in enumerate(q):
  ranks=np.argsort(-s[i],kind='stable')
  source_id=x['id'].split('|')[0];gold_id=idx[source_id]
  top1=int(int(ranks[0]) in gold[i] and s[i,ranks[0]]>0)
  top5=int(any(int(z) in gold[i] and s[i,z]>0 for z in ranks[:5]))
  top10=int(any(int(z) in gold[i] and s[i,z]>0 for z in ranks[:10]))
  exact=int(int(ranks[0])==gold_id and s[i,ranks[0]]>0)
  rows.append({'id':x['id'],'source':x['source'],'issue':x['issue'],'out_training_issue':x['out_training_issue'],'lid_correct':x['correct'],'multi':int(len(gold[i])>1),'source_page':source_id,'any1':top1,'any5':top5,'any10':top10,'exact1':exact,'num_gold':len(gold[i]),'method':name})
 res.extend(rows);summ[name]={}
 for subset,g in [('all',rows),('IMVO',[r for r in rows if r['source']=='IMVO']),('ISIG',[r for r in rows if r['source']=='ISIG']),('unique',[r for r in rows if not r['multi']]),('out_training_issues',[r for r in rows if r['out_training_issue']]),('LID_wrong',[r for r in rows if not r['lid_correct']])]:
  if g:summ[name][subset]={'n':len(g),**{k:float(np.mean([r[k] for r in g])) for k in ('any1','any5','any10','exact1')}}
 print(name,'all',summ[name]['all'],'IMVO',summ[name]['IMVO'],'ISIG',summ[name]['ISIG'],flush=True)
(P/'minority_retrieval_scores.npz').write_bytes(b'') if False else None
with (P/'minority_retrieval_results.jsonl').open('w') as f:
 for r in res:f.write(json.dumps(r)+'\n')
(P/'minority_retrieval_summary.json').write_text(json.dumps(summ,indent=2))
# Paired grouped issue cluster bootstrap.
rng=np.random.default_rng(20261023)
methods={m:{r['id']:r for r in res if r['method']==m} for m in summ}
issues={source:sorted({x['issue'] for x in q if x['source']==source}) for source in ('IMVO','ISIG')}
qids_by_issue=defaultdict(list)
for x in q:qids_by_issue[x['issue']].append(x['id'])
def paired(a,b,subset=None,iterations=5000):
 cohort=[z for z in q if subset is None or z['source']==subset]
 group={s:[x for x in issues[s] if any(y['issue']==x for y in cohort)] for s in issues}
 vals=np.array([methods[a][z['id']]['any1']-methods[b][z['id']]['any1'] for z in cohort]);boot=[]
 valid=set(x['id'] for x in cohort)
 for i in range(iterations):
  ids=[]
  for s,iv in group.items():
   if iv:
    selected=rng.choice(iv,size=len(iv),replace=True)
    ids.extend(x for iss in selected for x in qids_by_issue[iss] if x in valid)
  boot.append(float(np.mean([methods[a][x]['any1']-methods[b][x]['any1'] for x in ids])))
 return {'point':float(vals.mean()),'ci95':[float(x) for x in np.quantile(boot,[.025,.975])],'n':len(cohort)}
unc={}
for s in (None,'IMVO','ISIG'):
 unc['auto_minus_oracle_'+str(s)]=paired('auto','oracle_en',s)
 unc['oracle_minus_raw_'+str(s)]=paired('oracle_en','raw',s)
 unc['auto_minus_raw_'+str(s)]=paired('auto','raw',s)
(P/'minority_retrieval_uncertainty.json').write_text(json.dumps(unc,indent=2))
