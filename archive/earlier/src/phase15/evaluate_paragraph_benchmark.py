"""Controlled paragraph-faithful retrieval analysis, no model tuning."""
from pathlib import Path
import pickle,json,collections,time,gc,sys,numpy as np,scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import regex
ROOT=Path('/mnt/data/isiXhosa_three_track_pilot'); O=Path('/mnt/data/isiXhosa_phase15_source_grounded_search');t0=time.time()
D=pickle.load(open(ROOT/'corpus_cache.pkl','rb'));Q=[json.loads(s) for s in open(O/'paragraph_queries.jsonl')];N=len(D);M=len(Q)
idx={d['id']:i for i,d in enumerate(D)}
qtexts=[q['query'] for q in Q]+[q['noisy_query'] for q in Q]
S={}
for field in ('raw','xh','en'):
 print('FIT BM25',field,flush=True)
 v=CountVectorizer(lowercase=True,token_pattern=r'(?u)\b[^\W\d_]+\b',dtype=np.float32)
 X=v.fit_transform([d[field] for d in D]).tocsr()
 lens=np.asarray(X.sum(axis=1)).ravel().astype(np.float32);df=np.asarray((X>0).sum(axis=0)).ravel();idf=np.log1p((N-df+.5)/(df+.5));mean=max(float(lens.mean()),1.)
 co=X.tocoo();weights=co.data*2.2/(co.data+1.2*(.25+.75*lens[co.row]/mean))*idf[co.col]
 W=sp.csr_matrix((weights,(co.row,co.col)),shape=co.shape,dtype=np.float32)
 T=v.transform(qtexts);T.data[:]=1
 scores=(T@W.T).toarray().astype(np.float32);S[field]=scores
 del X,v,co,W,T,weights,scores;gc.collect()
 print(' DONE',field,round(time.time()-t0,1),flush=True)
# Check character ngrams as an independent fuzzy lexical baseline
print('FIT character TFIDF',flush=True)
c=TfidfVectorizer(analyzer='char',ngram_range=(3,4),max_features=85000,min_df=2,sublinear_tf=True,dtype=np.float32)
X=c.fit_transform([d['raw'] for d in D]);Y=c.transform(qtexts);S['char_tfidf']=(Y@X.T).toarray().astype(np.float32)
del X,Y,c;gc.collect()
print(' DONE char',round(time.time()-t0,1),flush=True)
# Fixed RRF reciprocal 60 within raw BM25 and character scores
rawrank=np.argsort(-S['raw'],axis=1,kind='stable');charrank=np.argsort(-S['char_tfidf'],axis=1,kind='stable')
posraw=np.empty_like(rawrank);poschar=np.empty_like(charrank)
for i in range(len(qtexts)):
 posraw[i,rawrank[i]]=np.arange(1,N+1,dtype=np.int32)
 poschar[i,charrank[i]]=np.arange(1,N+1,dtype=np.int32)
S['rrf']=1/(60+posraw.astype(np.float32))+1/(60+poschar.astype(np.float32))
# Language router, identical setup to Phase 15 precommit/Phase 14 pilot
wpat=regex.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
train=[];labels=[];early_issues=set()
for d in D:
 if d['source']!='IMVO' or not (1884<=d['year']<=1887):continue
 early_issues.add(d['issue'])
 for language in ('xh','en'):
  w=wpat.findall(d[language]);
  if len(w)<60:continue
  for i in [max(0,len(w)//4-30),max(0,3*len(w)//4-30) if len(w)>=250 else None]:
   if i is not None:train.append(' '.join(w[i:i+60]));labels.append(int(language=='xh'))
f=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,max_features=80000,sublinear_tf=True,dtype=np.float32)
v=f.fit_transform(train);lid=LogisticRegression(C=2.,class_weight='balanced',solver='liblinear',max_iter=300,random_state=20261023).fit(v,labels)
p=lid.predict_proba(f.transform(qtexts))[:,1];route=p>=.5; actual=np.array([q['lang']=='xh' for q in Q]*2)
S['fielded_oracle']=np.where(actual[:,None],S['xh'],S['en'])
S['fielded_auto']=np.where(route[:,None],S['xh'],S['en'])
np.savez_compressed(O/'paragraph_scores.npz',**S)
(O/'query_lid_accuracy.json').write_text(json.dumps({'clean':float(np.mean(route[:M]==actual[:M])), 'noisy':float(np.mean(route[M:]==actual[M:])),'clean_out_of_training_issues':float(np.mean([route[i]==actual[i] for i,q in enumerate(Q) if q['issue'] not in early_issues]))},indent=2))
# Evaluate rankings for all methods and both conditions; no page filtering
res=[]; summary={}
for method,scores in S.items():
 summary[method]={};ranks=np.argsort(-scores,axis=1,kind='stable');
 for kind,off in [('clean',0),('noisy',M)]:
  out=[]
  for i,q in enumerate(Q):
   row=ranks[i+off];rmap=np.empty(N,dtype=np.int32);rmap[row]=np.arange(1,N+1,dtype=np.int32)
   gold=idx[q['page']];goods=np.array([idx[x] for x in q['match_pages']],dtype=np.int32)
   # For empty field no lexical evidence, return no match, don't credit deterministic tie
   goldrank=int(rmap[gold]) if scores[i+off,gold]>0 else N+1
   valid=goods[scores[i+off,goods]>0];anyrank=int(rmap[valid].min()) if len(valid) else N+1
   rec=dict(qid=q['id'],source=q['source'],lang=q['lang'],issue=q['issue'],page=q['page'],page_num=q['source_page_num'],
      multi=int(q['num_exact_match_pages']>1),repeat_ge10=int(q['num_exact_match_pages']>=10),
      golds=q['num_exact_match_pages'],exact_rank=goldrank,any_rank=anyrank,exact1=int(goldrank==1),
      any1=int(anyrank==1),exact5=int(goldrank<=5),any5=int(anyrank<=5),method=method,kind=kind,
      route_correct=int(route[i+off]==actual[i+off]))
   res.append(rec);out.append(rec)
  for k,group in [('all',out),('nonfront',[r for r in out if r['page_num']>1]),('unique',[r for r in out if not r['multi']]),
   ('nonfront_unique',[r for r in out if r['page_num']>1 and not r['multi']]),('nonfront_nonhighrepeat',[r for r in out if r['page_num']>1 and not r['repeat_ge10']])]+[(source+'_'+lang,[r for r in out if r['source']==source and r['lang']==lang]) for source in ('IMVO','ISIG') for lang in ('xh','en')]:
    if not group:continue
    summary[method][kind+'_'+k]={'n':len(group),'exact1':round(float(np.mean([r['exact1'] for r in group])),6),
      'any1':round(float(np.mean([r['any1'] for r in group])),6),
      'exact5':round(float(np.mean([r['exact5'] for r in group])),6),
      'any5':round(float(np.mean([r['any5'] for r in group])),6),
      'repeat':round(float(np.mean([r['multi'] for r in group])),6)}
 print('RESULT',method,'clean_all',summary[method]['clean_all'],'noisy_all',summary[method]['noisy_all'],flush=True)
with (O/'paragraph_retrieval_results.jsonl').open('w') as f:
 for o in res:f.write(json.dumps(o)+'\n')
(O/'paragraph_retrieval_summary.json').write_text(json.dumps(summary,indent=2))
print('seconds',round(time.time()-t0,1),flush=True)
