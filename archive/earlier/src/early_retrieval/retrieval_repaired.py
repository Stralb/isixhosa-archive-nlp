"""Repaired source-verbatim benchmark, multi-source ambiguity, and dual-view language-aware BM25."""
from pathlib import Path
import pickle,json,collections,time,gc,numpy as np,scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
O=Path('/mnt/data/isiXhosa_three_track_pilot');t0=time.time()
with (O/'corpus_cache.pkl').open('rb') as f:docs=pickle.load(f)
Q=[json.loads(x) for x in (O/'query_repaired.jsonl').open()]
N=len(docs);M=len(Q); idx={d['id']:i for i,d in enumerate(docs)}
queries=[q['query'] for q in Q]+[q['noisy_query'] for q in Q]
# Fit separate unrestricted BM25 field indexes with shared exact source-page IDs.
def fit_score(field):
 print('FIT',field,flush=True)
 c=CountVectorizer(lowercase=True,token_pattern=r'(?u)\b[^\W\d_]+\b',dtype=np.float32)
 X=c.fit_transform([r[field] for r in docs]).tocsr()
 lens=np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
 dfs=np.asarray((X>0).sum(axis=0)).ravel()
 idf=np.log1p((N-dfs+.5)/(dfs+.5)); mean=max(float(lens.mean()),1)
 co=X.tocoo()
 data=co.data*2.2/(co.data+1.2*(.25+.75*lens[co.row]/mean))*idf[co.col]
 W=sp.csr_matrix((data,(co.row,co.col)),shape=co.shape,dtype=np.float32)
 T=c.transform(queries);T.data[:]=1
 S=(T@W.T).toarray().astype(np.float32)
 del X,c,co,W,T;gc.collect()
 print('DONE field',field,'elapsed',round(time.time()-t0,1),flush=True)
 return S
raw=fit_score('raw');xh=fit_score('xh');en=fit_score('en')
np.savez_compressed(O/'repaired_scores.npz',raw=raw,xh=xh,en=en)
# Per-query oracle-known-language fielded retrieval, an optimistic ceiling.
oracle=np.where(np.array([q['lang']=='xh' for q in Q]*2)[:,None],xh,en)
# Evaluate source language routing using an IMVO 1884-87 training-only classifier: query-scale generalization is separate.
import regex as rx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
WORD=rx.compile(r'(?V1)\p{L}+(?:[\'’]\p{L}+)*')
train=[];y=[];early_issues=set()
for d in docs:
 if d['source']!='IMVO' or d['year']<1884 or d['year']>1887:continue
 early_issues.add(d['issue'])
 for lang in ('xh','en'):
  w=WORD.findall(d[lang])
  if len(w)<60:continue
  for i in [max(0,len(w)//4-30),max(0,3*len(w)//4-30) if len(w)>=250 else None]:
   if i is not None: train.append(' '.join(w[i:i+60]));y.append(int(lang=='xh'))
v=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,max_features=80000,sublinear_tf=True,dtype=np.float32)
tr=v.fit_transform(train);lm=LogisticRegression(C=2.,class_weight='balanced',solver='liblinear',max_iter=300,random_state=20261023).fit(tr,y)
prob=lm.predict_proba(v.transform(queries))[:,1]; route=prob>=.5
true=np.array([q['lang']=='xh' for q in Q]*2)
np.savez_compressed(O/'query_lid_route.npz',prob=prob,route=route,true=true)
auto=np.where(route[:,None],xh,en)

def ranks(S):return np.argsort(-S,axis=1,kind='stable')
def eligible_rank(R,mask):
 result=np.empty_like(R)
 for j,r in enumerate(R):result[j]=np.concatenate([r[mask[r]],r[~mask[r]]])
 return result
rankmaps={'raw':ranks(raw),'Xh_only':ranks(xh),'En_only':ranks(en),'fielded_oracle':ranks(oracle),'fielded_auto_LID':ranks(auto)}
rxraw=rankmaps['raw'];
for lang in ('xh','en'):
 if lang=='xh':
  major=np.array([d['xh_pct']>=50 for d in docs]); tags=np.array([int(d['tracker_xh_words'] or 0)>=50 for d in docs])
 else:
  major=np.array([d['en_pct']>=50 for d in docs]); tags=np.array([int(d['tracker_en_words'] or 0)>=50 for d in docs])
 rankmaps[f'raw_filter_{lang}_majority']=(eligible_rank(rxraw,major),major)
 rankmaps[f'raw_filter_{lang}_multitag']=(eligible_rank(rxraw,tags),tags)

res=[]
for method,rp in rankmaps.items():
 R,mask=rp if isinstance(rp,tuple) else (rp,None)
 for i,q in enumerate(Q):
  good=np.array([idx[g] for g in q['match_pages']],dtype=np.int32)
  gold=idx[q['page']]
  for kind,j in [('clean',i),('noisy',M+i)]:
   row=R[j]; rankarr=np.empty(N,dtype=np.int32);rankarr[row]=np.arange(1,N+1)
   if mask is None:
    rank_gold=int(rankarr[gold]); valid_good=good
   else:
    rank_gold=int(rankarr[gold]) if mask[gold] else N+1
    valid_good=good[mask[good]]
   # Empty text fields must never be credited with matching queries just through tie order.
   score_matrix = {'raw':raw,'Xh_only':xh,'En_only':en,'fielded_oracle':oracle,'fielded_auto_LID':auto}.get(method)
   if score_matrix is not None:
    if score_matrix[j,gold]<=0:rank_gold=N+1
    valid_good=valid_good[score_matrix[j,valid_good]>0]
   rank_any=int(np.min(rankarr[valid_good])) if len(valid_good)>0 else N+1
   res.append(dict(method=method,kind=kind,qid=q['id'],source=q['source'],lang=q['lang'],issue=q['issue'],page=q['page'],
       originally_repaired=int(q['repaired']),multi_gold=int(q['num_exact_match_pages']>1),gold_count=q['num_exact_match_pages'],
       exact_rank=rank_gold,any_rank=rank_any,exact_r1=int(rank_gold==1),exact_r5=int(rank_gold<=5),
       any_r1=int(rank_any==1),any_r5=int(rank_any<=5),
       page_present=int(mask is None or mask[gold]),
       q_lang_correct=int(route[j]==true[j])))
with (O/'retrieval_repaired_results.jsonl').open('w') as f:
 for o in res:f.write(json.dumps(o)+'\n')
S={}
for m in rankmaps:
 S[m]={}
 for kind in ('clean','noisy'):
  for group,items in [('all',[r for r in res if r['method']==m and r['kind']==kind])]+[(src+'_'+lang,[r for r in res if r['method']==m and r['kind']==kind and r['source']==src and r['lang']==lang]) for src in ('IMVO','ISIG') for lang in ('xh','en')]:
   if not items:continue
   S[m][kind+'_'+group]={'n':len(items),**{k:round(float(np.mean([r[k] for r in items])),6) for k in ('exact_r1','exact_r5','any_r1','any_r5','page_present','q_lang_correct')},'multi_gold_fraction':round(float(np.mean([r['multi_gold'] for r in items])),6)}
(O/'retrieval_repaired_summary.json').write_text(json.dumps(S,indent=2))
for m in rankmaps:
 print('\n',m,flush=True)
 for name,vals in S[m].items():
  if name.endswith('all') or name in ('clean_IMVO_xh','clean_IMVO_en','clean_ISIG_xh','clean_ISIG_en','noisy_IMVO_xh','noisy_IMVO_en'):
   print(' ',name,'N',vals['n'],'exact1',vals['exact_r1'],'any1',vals['any_r1'],'exact5',vals['exact_r5'],'cover',vals['page_present'],flush=True)
print('auto LID accuracy',{'all_clean':float(np.mean(route[:M]==true[:M])),'all_noisy':float(np.mean(route[M:]==true[M:])),
  'historically_heldout_clean':float(np.mean([route[i]==true[i] for i,q in enumerate(Q) if q['issue'] not in early_issues]))},flush=True)
print('FINISHED',round(time.time()-t0,1),flush=True)
