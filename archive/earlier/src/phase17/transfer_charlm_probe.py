from pathlib import Path
import pickle,json,hashlib,re,math,random,collections,functools,time
import numpy as np
P=Path('/mnt/data/isiXhosa_phase17_deeper_questions')
D=pickle.load(open('/mnt/data/isiXhosa_three_track_pilot/corpus_cache.pkl','rb'))
modern=[json.loads(x) for x in open('/mnt/data/xho.jsonl') if x.strip()]
# Cross-source original splits; 1888 was inspected in earlier phases and remains retrospective.
hist_train=[x for x in D if x['source'] in ('IMVO','ISIG') and x['year']==1887 and x['date'][:7]<'1887-10' and len(x['xh'])>200]
hist_test=[x for x in D if x['source'] in ('IMVO','ISIG') and x['year']==1888 and len(x['xh'])>300]
def hashbucket(x):return int(hashlib.sha256(x.encode()).hexdigest()[:12],16)%4
modern_train=[{'id':f'XHO_{i}','issue':f'XHO_{i}','source':'MODERN','xh':r.get('text','') or ''} for i,r in enumerate(modern) if hashbucket(f'modern_{i}'+r.get('title',''))<3 and len(r.get('text','') or '')>100]
modern_test=[{'id':f'XHO_{i}','issue':f'XHO_{i}','source':'MODERN','xh':r.get('text','') or ''} for i,r in enumerate(modern) if hashbucket(f'modern_{i}'+r.get('title',''))>=3 and len(r.get('text','') or '')>100]
# Define normalization once for all domains; spaces preserved, punctuation not modelled.
normalize_re=re.compile(r'[^a-z]+')
def norm(s):return normalize_re.sub(' ',s.lower()).strip()+' '
hist_train=[{**d,'seq':norm(d['xh'])} for d in hist_train];modern_train=[{**d,'seq':norm(d['xh'])} for d in modern_train]
hist_test=[{**d,'seq':norm(d['xh'])} for d in hist_test];modern_test=[{**d,'seq':norm(d['xh'])} for d in modern_test]
assert len(modern_test)>25 and len(modern_train)>75
print('sizes',len(hist_train),len(hist_test),len(modern_train),len(modern_test),flush=True)
BUDGET=200000; ALPH='abcdefghijklmnopqrstuvwxyz '; ORDER=4; GAMMA=20.;V=len(ALPH)
# Sampling whole documents by reproducible shuffle, final train doc clipped to fixed budget.
def sample(pool,budget,seed):
 rng=random.Random(seed);pool=list(pool);rng.shuffle(pool)
 out=[];remaining=budget
 for doc in pool:
  if remaining<=0:break
  s=doc['seq']
  if len(s)>remaining:s=s[:remaining]
  out.append(s);remaining-=len(s)
 if remaining>0:raise RuntimeError(f'Not enough data: {remaining}')
 return out
class InterpolatedCharLM:
 def __init__(self,strings):
  self.cnt=[collections.Counter() for _ in range(ORDER+1)]
  self.tot=[collections.Counter() for _ in range(ORDER+1)]
  for seq in strings:
   for i,ch in enumerate(seq):
    self.cnt[0][ch]+=1;self.tot[0]['']+=1
    for o in range(1,ORDER+1):
     if i<o:break
     ctx=seq[i-o:i];self.cnt[o][ctx+ch]+=1;self.tot[o][ctx]+=1
  c=self.cnt;t=self.tot
  @functools.lru_cache(maxsize=250000)
  def ll(gram):
   p=(c[0][gram[-1]]+.01)/(t[0]['']+.01*V)
   for o in range(1,ORDER+1):
    ctx=gram[ORDER-o:ORDER];den=t[o][ctx]
    if den:p=(c[o][ctx+gram[-1]]+GAMMA*p)/(den+GAMMA)
   return -math.log2(max(p,1.e-20))
  self.ll=ll
 def evaluate(self,seq):
  if len(seq)<=ORDER:return (0.,0)
  a=sum(self.ll(seq[i-ORDER:i+1]) for i in range(ORDER,len(seq)))
  return (a,len(seq)-ORDER)
# Eval fixed 1200-char windows per historical page, 1500 per modern article. Each individual page remains a cluster.
def evalwindow(doc,limit):
 s=doc['seq'];hashcode=int(hashlib.sha256(doc['id'].encode()).hexdigest()[:12],16)
 if len(s)<=limit+ORDER:return s
 maxstart=len(s)-(limit+ORDER);start=hashcode%(maxstart+1)
 return s[start:start+limit+ORDER]
E=[{**d,'eval_seq':evalwindow(d,1200)} for d in hist_test]+[{**d,'eval_seq':evalwindow(d,1500)} for d in modern_test]
train_info={'historical_1887_pages':len(hist_train),'historical_1888_test_pages':len(hist_test),'modern_training_articles':len(modern_train),'modern_test_articles':len(modern_test),'hist_train_chars':sum(len(d['seq']) for d in hist_train),'modern_train_chars':sum(len(d['seq']) for d in modern_train),'budget_chars':BUDGET,'model':'interpolated fifth-order character n-gram LM, gamma20, alphabet a-z and space','provenance':'previously inspected corpora: retrospective transfer diagnostic; not confirmation'}
(P/'transfer_data_audit.json').write_text(json.dumps(train_info,indent=2))
res=[]; t0=time.time();seeds=[20261023,20261024,20261025]
for seed in seeds:
 groups={'HIST_ONLY':sample([d for d in hist_train if d['source']=='IMVO'],BUDGET//2,seed)+sample([d for d in hist_train if d['source']=='ISIG'],BUDGET//2,seed+100),
         'MODERN_ONLY':sample(modern_train,BUDGET,seed+200),
         'HALF_HALF':sample([d for d in hist_train if d['source']=='IMVO'],BUDGET//4,seed)+sample([d for d in hist_train if d['source']=='ISIG'],BUDGET//4,seed+100)+sample(modern_train,BUDGET//2,seed+200)}
 for method,train_strings in groups.items():
  model=InterpolatedCharLM(train_strings)
  for d in E:
   loss,n=model.evaluate(d['eval_seq']);res.append({'method':method,'seed':seed,'source':d['source'],'issue':d['issue'],'page':d['id'],'total_bits':round(loss,9),'chars':n,'bpc':loss/n if n else None})
  print('model',method,'seed',seed,'duration',round(time.time()-t0,2),flush=True)
with (P/'transfer_charlm_results.jsonl').open('w') as f:
 for row in res:f.write(json.dumps(row)+'\n')
report={}
for method in ('HIST_ONLY','MODERN_ONLY','HALF_HALF'):
 report[method]={}
 for source in ('IMVO','ISIG','MODERN'):
  vals=[]
  for seed in seeds:
   rows=[r for r in res if r['method']==method and r['source']==source and r['seed']==seed]
   vals.append(sum(r['total_bits'] for r in rows)/sum(r['chars'] for r in rows))
  report[method][source]={'seeds_bpc':vals,'mean_bpc':sum(vals)/len(vals),'sd_seeds':float(np.std(vals,ddof=1))}
 report[method]['hist_macro']=(report[method]['IMVO']['mean_bpc']+report[method]['ISIG']['mean_bpc'])/2
 report[method]['worst_of_3']=max(report[method][x]['mean_bpc'] for x in ('IMVO','ISIG','MODERN'))
 report[method]['macro_of_3']=sum(report[method][x]['mean_bpc'] for x in ('IMVO','ISIG','MODERN'))/3
(P/'transfer_charlm_summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
print('TOTAL',time.time()-t0,flush=True)
