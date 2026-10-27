#!/usr/bin/env python3
"""Phase30 precommitted exact-form analysis and source-blind BPE placebos."""
from __future__ import annotations
import collections, csv, hashlib, json, math, os, random
from pathlib import Path
import numpy as np
import sentencepiece as spm
from scipy.stats import rankdata, spearmanr

D=Path(__file__).resolve().parent; P=Path(os.environ.get('PHASE30_INPUTS',str(D/'inputs')))
W=[json.loads(line) for line in (P/'windows.jsonl').open(encoding='utf8')]
T=[r for r in W if r['year']==1888]
assert len(W)==581 and len(T)==207 and all(len(r['tokens'])==90 for r in T)
SOURCES=('IMVO','ISIG'); SEEDS=range(3)

def chars(t):return sum(ch.isalpha() for ch in t)
def groupbin(n,bins):
 for i,u in enumerate(bins):
  if n<=u:return i
 return len(bins)
def scsv(name,rows):
 if not rows:return
 with (D/name).open('w',encoding='utf8',newline='') as f:
  v=csv.DictWriter(f,fieldnames=list(rows[0]));v.writeheader();v.writerows(rows)

testcnt={s:collections.Counter(t for w in T if w['source']==s for t in w['tokens']) for s in SOURCES}
shared=set(testcnt['IMVO']) & set(testcnt['ISIG'])
test_issues={s:sorted(set(w['issue'] for w in T if w['source']==s)) for s in SOURCES}
models={(s,k):spm.SentencePieceProcessor(model_file=str(P/f'BPE_{s}_seed{k}.model')) for s in SOURCES for k in SEEDS}
caches={key:{} for key in models}
def count_model(key,t):
 if t not in caches[key]:caches[key][t]=len(models[key].encode(t,out_type=int))
 return caches[key][t]

traincnt={(s,k):collections.Counter((P/f'train_{s}_seed{k}.txt').read_text(encoding='utf8').split()) for s in SOURCES for k in SEEDS}
# Confirm training characters equal budget
train_audit=[]
for k in SEEDS:
 for s in SOURCES:
  train_chars=sum(c*chars(t) for t,c in traincnt[s,k].items())
  train_audit.append(dict(source=s,seed=k,alphabetic_characters=train_chars,word_tokens=sum(traincnt[s,k].values()),word_types=len(traincnt[s,k])))
  assert train_chars==60000,(s,k,train_chars)
scsv('frozen_training_audit.csv',train_audit)

by_seed=[];sampleout=[];eligible_words={};cue_roots={'imvo','isigidimi','xosa','jabavu','bokwe','gqoba','makiwane','hofmeyr'}

def spearman_permutation(rows,k):
 x=np.array([r['train_frequency_log_ratio'] for r in rows]);y=np.array([r['isig_minus_imvo_pieces'] for r in rows]);
 rx=rankdata(x,method='average');ry=rankdata(y,method='average');
 rx=(rx-rx.mean())/(rx.std()+1e-15);ry=(ry-ry.mean())/(ry.std()+1e-15)
 obs=float((rx*ry).mean())
 grouped=collections.defaultdict(list)
 for j,r in enumerate(rows):
  a=groupbin(r['word_alpha_length'],[3,6,9,12]);b=groupbin(r['training_frequency_total'],[3,7,15]);grouped[(a,b)].append(j)
 rng=np.random.default_rng(30024+k);null=[]
 for _ in range(2000):
  z=rx.copy()
  for inds in grouped.values():
   if len(inds)>=2:
    ids=np.asarray(inds);z[ids]=rx[rng.permutation(ids)]
  null.append(float((z*ry).mean()))
 p=(1+sum(v>=obs-1e-12 for v in null))/(len(null)+1)
 rngb=np.random.default_rng(30624+k);boots=[]
 for _ in range(1000):
  j=rngb.integers(0,len(x),size=len(x));boots.append(float(spearmanr(x[j],y[j]).statistic))
 ci=list(map(float,np.quantile(boots,[.025,.975])))
 return dict(rho=obs,p_stratified_permutation=float(p),type_bootstrap_95ci=ci,null95=float(np.quantile(null,.95)),n_types=len(rows),n_nontie_pieces=int(sum(t!=0 for t in y)))

for k in SEEDS:
 rows=[]
 for t in sorted(shared):
  a=traincnt['IMVO',k][t];b=traincnt['ISIG',k][t]
  if a+b<2:continue
  c_i=count_model(('IMVO',k),t);c_s=count_model(('ISIG',k),t)
  rows.append(dict(seed=k,word=t,word_alpha_length=chars(t),train_IMVO=a,train_ISIG=b,
     training_frequency_total=a+b,train_frequency_log_ratio=float(math.log((a+.5)/(b+.5))),
     pieces_IMVO=c_i,pieces_ISIG=c_s,isig_minus_imvo_pieces=c_s-c_i,
     heldout_IMVO_occurrences=testcnt['IMVO'][t],heldout_ISIG_occurrences=testcnt['ISIG'][t]))
 sampleout+=rows;eligible_words[k]={r['word'] for r in rows}
 filtered=[r for r in rows if not any(c in r['word'] for c in cue_roots)]
 by_seed.append(dict(seed=k,all=spearman_permutation(rows,k),no_titles_editors=spearman_permutation(filtered,k)))
scsv('shared_word_frequency_vs_pieces.csv',sampleout)

# Test-window exact surface form decompositions, and strict source-model/page result reproduction
ref={(r['file'],int(r['seed'])):r for r in csv.DictReader((P/'tokenizer_page_scores.csv').open(encoding='utf8'))}
page_rows=[];pagecells=collections.defaultdict(lambda:np.zeros(4,dtype=np.int64)) # all chars, all saving, shared chars, shared saving
max_err=0
for w in T:
 for k in SEEDS:
  vals={s:sum(count_model((s,k),t) for t in w['tokens']) for s in SOURCES}
  rec=ref[w['file'],k];nchar=sum(chars(t) for t in w['tokens'])
  for key,v in [('IMVO_pieces',vals['IMVO']),('ISIG_pieces',vals['ISIG']),('alphabetic_chars',nchar)]:max_err=max(max_err,abs(float(rec[key])-v))
  assert nchar>0
  sign=1 if w['source']=='IMVO' else -1
  benefit=sign*(vals['ISIG']-vals['IMVO'])
  words=[t for t in w['tokens'] if t in shared]
  sharedchar=sum(chars(t) for t in words)
  sharedbenefit=sign*sum(count_model(('ISIG',k),t)-count_model(('IMVO',k),t) for t in words)
  pagecells[w['source'],k,w['issue']]+=np.array([nchar,benefit,sharedchar,sharedbenefit])
  page_rows.append(dict(file=w['file'],issue=w['issue'],source=w['source'],seed=k,all_alphabetic_chars=nchar,
      all_specialist_saved_pieces=benefit,shared_alphabetic_chars=sharedchar,shared_specialist_saved_pieces=sharedbenefit,
      exclusive_alphabetic_chars=nchar-sharedchar,exclusive_specialist_saved_pieces=benefit-sharedbenefit))
assert max_err==0,max_err
scsv('source_page_shared_decomposition.csv',page_rows)

def group_metrics(cells,s,k):
 ss=[i for (ss,kk,i) in cells if ss==s and kk==k];v=sum((cells[s,k,i] for i in ss),np.zeros(4,dtype=float))
 return dict(source=s,seed=k,n_issues=len(ss),all_efficiency_advantage=100*v[1]/v[0],shared_efficiency_advantage=100*v[3]/v[2],
   char_share_shared=v[2]/v[0],piece_saving_share_shared=(v[3]/v[1] if v[1] else None),all_saved_pieces=int(v[1]),shared_saved_pieces=int(v[3]))
source_metrics=[group_metrics(pagecells,s,k) for s in SOURCES for k in SEEDS]
scsv('source_seed_decomposition.csv',source_metrics)

# Placebo input generation: shuffle pooled train tokens, assign alternating to less-filled, trim to exact common alphabetic budget.
def trim_to(tokens,target):
 cum=0;prefix=[]
 for t in tokens:
  if cum>=target:break
  prefix.append(t);cum+=chars(t)
 if cum==target:return prefix
 dif=cum-target
 # Drop a small deterministic subset near the tail whose char lengths sum to exactly overshoot.
 tails=prefix[-100:];dp={0:()}
 for j,t in enumerate(tails):
  length=chars(t)
  for x,ix in list(dp.items())[::-1]:
   new=x+length
   if new<=dif and new not in dp:dp[new]=ix+(j,)
  if dif in dp:break
 if dif not in dp:raise RuntimeError(f'Cannot trim to common exact char budget; overshoot {dif}')
 idx=set(dp[dif]);keep=prefix[:-100]+[t for j,t in enumerate(tails) if j not in idx] if len(prefix)>=100 else [t for j,t in enumerate(tails) if j not in idx]
 assert sum(map(chars,keep))==target
 return keep

placebomodels={}
placebo_audit=[]
for k in SEEDS:
 pool=[]
 for s in SOURCES:pool+=list((P/f'train_{s}_seed{k}.txt').read_text(encoding='utf8').split())
 r=random.Random(30124+k);r.shuffle(pool)
 groups=[[],[]];tot=[0,0]
 for tok in pool:
  j=int(tot[1]<tot[0])
  groups[j].append(tok);tot[j]+=chars(tok)
 assert sum(tot)==120000
 budget=min(59950,*tot)
 for j in (0,1):groups[j]=trim_to(groups[j],budget)
 for j,code in enumerate(('A','B')):
  path=D/f'placebo_{code}_seed{k}.txt';path.write_text('\n'.join(' '.join(groups[j][i:i+90]) for i in range(0,len(groups[j]),90))+'\n',encoding='utf8')
  assert sum(chars(t) for t in groups[j])==budget
  prefix=str(D/f'PLACEBO_{code}_seed{k}')
  spm.SentencePieceTrainer.train(input=str(path),model_prefix=prefix,model_type='bpe',vocab_size=1000,character_coverage=1.0,normalization_rule_name='identity',hard_vocab_limit=False,unk_id=0,bos_id=-1,eos_id=-1,pad_id=-1,shuffle_input_sentence=False,minloglevel=2)
  placebomodels[code,k]=spm.SentencePieceProcessor(model_file=prefix+'.model')
  placebo_audit.append(dict(seed=k,group=code,n_alpha=budget,n_tokens=len(groups[j]),train_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
scsv('placebo_train_audit.csv',placebo_audit)

placebocache={(s,k):{} for s in ('A','B') for k in SEEDS}
def pc(key,t):
 if t not in placebocache[key]:placebocache[key][t]=len(placebomodels[key].encode(t,out_type=int))
 return placebocache[key][t]
placebocells=collections.defaultdict(lambda:np.zeros(2,dtype=np.int64));placebo_pages=[]
for w in T:
 for k in SEEDS:
  ch=sum(chars(t) for t in w['tokens'])
  costA=sum(pc(('A',k),t) for t in w['tokens']);costB=sum(pc(('B',k),t) for t in w['tokens'])
  sign=1 if w['source']=='IMVO' else -1
  sav=sign*(costB-costA)
  placebocells[w['source'],k,w['issue']]+=np.array([ch,sav])
  placebo_pages.append(dict(file=w['file'],source=w['source'],issue=w['issue'],seed=k,alphabetic_chars=ch,placebo_A_pieces=costA,placebo_B_pieces=costB,oriented_specialist_savings=sav))
scsv('placebo_page_scores.csv',placebo_pages)

def row_sum(cells,s,k,indices,dim):
 return sum((cells[s,k,iss] for iss in indices),np.zeros(dim,dtype=float))

def agg_original(k,s,kind='all'):
 v=row_sum(pagecells,s,k,test_issues[s],4)
 if kind=='shared':return 100*v[3]/v[2]
 return 100*v[1]/v[0]
def agg_placebo(k,s):
 v=row_sum(placebocells,s,k,test_issues[s],2)
 return 100*v[1]/v[0]
source_blind_test=[]
for k in SEEDS:
 obs=sum(agg_original(k,s) for s in SOURCES)
 plc=sum(agg_placebo(k,s) for s in SOURCES)
 source_blind_test.append(dict(seed=k,actual_S=obs,random_source_S=plc,actual_minus_placebo=obs-plc,
     IMVO_actual=agg_original(k,'IMVO'),ISIG_actual=agg_original(k,'ISIG'),IMVO_placebo=agg_placebo(k,'IMVO'),ISIG_placebo=agg_placebo(k,'ISIG')))
scsv('observed_vs_placebo.csv',source_blind_test)

# Hierarchical issue/seed bootstrap, source strata independently; paired models per seed
rng=np.random.default_rng(30324); bvals=[]
for b in range(2000):
 k=int(rng.integers(0,3));r={'seed':k}
 for s in SOURCES:
  issues=np.array(test_issues[s]);draw=rng.choice(issues,size=len(issues),replace=True)
  v=row_sum(pagecells,s,k,draw,4);u=row_sum(placebocells,s,k,draw,2)
  r[s+'_shared']=100*v[3]/v[2];r[s+'_all']=100*v[1]/v[0];r[s+'_placebo']=100*u[1]/u[0]
 r['S_diff']=r['IMVO_all']+r['ISIG_all']-r['IMVO_placebo']-r['ISIG_placebo']
 bvals.append(r)
intervals={key:list(map(float,np.quantile([r[key] for r in bvals],[.025,.975]))) for key in ('IMVO_shared','ISIG_shared','IMVO_all','ISIG_all','IMVO_placebo','ISIG_placebo','S_diff')}

stats={'frozen_input_sha256':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(P.glob('*'))},
 'eligible':{'test_windows':len(T),'test_issues':{s:len(test_issues[s]) for s in SOURCES},'shared_test_types':len(shared),'heldout_covered_tokens':{s:sum(testcnt[s][t] for t in shared) for s in SOURCES},'total_heldout_tokens':{s:sum(testcnt[s].values()) for s in SOURCES}},
 'H30A':by_seed,
 'H30B':{'source_seed_metrics':source_metrics,'issue_seed_bootstrap_95ci':{s:intervals[s+'_shared'] for s in SOURCES},'mean_specialist_savings_on_shared_forms':{s:float(np.mean([agg_original(k,s,'shared') for k in SEEDS])) for s in SOURCES}},
 'H30C':{'per_seed':source_blind_test,'mean_observed_S':float(np.mean([r['actual_S'] for r in source_blind_test])),'mean_abs_placebo_S':float(np.mean([abs(r['random_source_S']) for r in source_blind_test])),'paired_bootstrap_delta95':intervals['S_diff'],'controls':placebo_audit},
 'all_and_placebo_bootstrap_95ci':intervals,
 'verification':{'piece_additivity_vs_phase26_max_error':max_err,'n_page_seed_pairs':len(page_rows),'n_additivity_checks':2*len(page_rows)},
 'decision_gates':{
 'H30A':all(z['all']['rho']>=.10 and z['all']['p_stratified_permutation']<=.05 for z in by_seed),
 'H30B':all(intervals[s+'_shared'][0]>0 for s in SOURCES),
 'H30C':np.mean([r['actual_S'] for r in source_blind_test])>2*np.mean([abs(r['random_source_S']) for r in source_blind_test]) and intervals['S_diff'][0]>0}}
(D/'RESULTS.json').write_text(json.dumps(stats,indent=2),encoding='utf8')
print(json.dumps({'H30A':[(x['seed'],x['all']) for x in by_seed], 'H30B':stats['H30B']['mean_specialist_savings_on_shared_forms'],'H30C':{k:stats['H30C'][k] for k in ['mean_observed_S','mean_abs_placebo_S','paired_bootstrap_delta95']},'gates':stats['decision_gates']},indent=2))
