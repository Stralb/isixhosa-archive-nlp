#!/usr/bin/env python3
import collections, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent))
import phase11 as common
from common import iter_pages,forms,alpha_len
from classic_bpe_intervene import ClassicBPE
from train_reference import sample_exact
assert hashlib.sha256((P/'TEMPORAL_PRECOMMIT.md').read_bytes()).hexdigest()==(P/'TEMPORAL_PRECOMMIT.sha256').read_text().split()[0]
DATA=P/'TEMPORAL_CORPUS_MANIFEST.json';CP=P/'TEMPORAL_TRAIN_CHOICES.json';MODELS=P/'models';RESULT=P/'TEMPORAL_RESULTS.json'
MODELS.mkdir(exist_ok=True)
YEARS=[1889,1890]; BUDGET=200000
PLANS=[(y,s) for y in YEARS for s in range(8101+(y-1889)*100,8113+(y-1889)*100)]
def sha(b):return hashlib.sha256(b).hexdigest()
def get_data():
 train={};test={};audit={}
 for y in YEARS:
  t=list(iter_pages(years=[str(y)],months={f'{x:02d}' for x in range(1,10)}))
  v=list(iter_pages(years=[str(y)],months={'10','11','12'}))
  train[y]=[w for r in t if r['source']=='IMVO' for w in forms(r['text'])]
  test[y]=[w for r in v if r['source']=='IMVO' for w in forms(r['text'])]
  a=sum(map(alpha_len,train[y]));b=sum(map(alpha_len,test[y]))
  assert a>=BUDGET and b>10000,(y,a,b)
  audit[y]={'train_alpha_chars':a,'eval_alpha_chars':b,'train_pages':sum(r['source']=='IMVO' for r in t),
            'eval_pages':sum(r['source']=='IMVO' for r in v),
            'train_words':len(train[y]),'eval_words':len(test[y]),
            'train_sha256':sha('\n'.join(train[y]).encode())}
 manifest={'years':{str(k):v for k,v in audit.items()},'sha':(P/'TEMPORAL_PRECOMMIT.sha256').read_text().split()[0]}
 if DATA.exists():
  assert json.loads(DATA.read_text())==manifest,'Temporal input source changed after manifest'
 else:
  DATA.write_text(json.dumps(manifest,indent=2))
 return train,test

def run_train(a,b):
 train,_=get_data()
 rows=json.loads(CP.read_text())['rows'] if CP.exists() else []
 assert len(rows)==a,(len(rows),a)
 for i in range(a,b):
  year,seed=PLANS[i]
  sample=sample_exact(train[year],BUDGET,seed*101+1)
  top,margin,z=common.first_pairs(sample)
  cand=[]
  for name,pair in [('greedy',top[0][0]),('second',top[1][0])]:
   t=time.monotonic();mod=ClassicBPE.fit(sample,vocab_size=2000,forced_prefix=[pair]); assert mod.merges[0]==pair
   file=f'{year}_{seed}_{name}.json';path=MODELS/file
   path.write_text(json.dumps(mod.to_json(),ensure_ascii=False))
   cand.append({'name':name,'file':file,'train_E':100*mod.train_token_count/BUDGET,
           'seconds':round(time.monotonic()-t,3),'sha256':sha(path.read_bytes())})
  choice=(min(cand,key=lambda z:(z['train_E'],z['name'])) if margin<=.02 else cand[0])
  rows.append({'year':year,'seed':seed,'sample_sha256':sha('\n'.join(sample).encode()),
       'top_pairs':[[list(pair),cnt] for pair,cnt in top],
       'margin':margin,'flag':margin<=.02,'z_approx':z,
       'candidates':cand,'selected':choice['name'],'selected_file':choice['file']})
  CP.write_text(json.dumps({'protocol_sha':(P/'TEMPORAL_PRECOMMIT.sha256').read_text().split()[0],
                'training_only':True,'rows':rows},indent=2,ensure_ascii=False))
  print('TEMP_TRAIN',i+1,'of',len(PLANS),year,seed,'first',top[0][0],top[1][0],
        'margin',round(margin,4),'flag',margin<=.02,'selected',choice['name'],flush=True)
 print('TEMP_TRAIN_CHUNK_DONE',b)

def full():
 train,_=get_data()
 for y in YEARS:
  path=MODELS/f'{y}_fullpool.json'
  if not path.exists():
   model=ClassicBPE.fit(train[y],vocab_size=2000)
   path.write_text(json.dumps(model.to_json(),ensure_ascii=False))
   print('TEMP_FULL',y,sha(path.read_bytes()))

def score():
 train,test=get_data();rows=json.loads(CP.read_text())['rows']
 assert len(rows)==len(PLANS)
 out={};rng=np.random.default_rng(20261023)
 for y in YEARS:
  rr=[r for r in rows if r['year']==y]
  raw=collections.Counter(test[y]);words=list(raw)
  known=set(train[y]);unseen=np.array([w not in known for w in words]);freq=np.array([raw[w] for w in words]);lengths=np.array([alpha_len(w) for w in words]);
  models={}
  for row in rr:
   for c in row['candidates']:
    path=MODELS/c['file'];assert sha(path.read_bytes())==c['sha256']
    models[c['file']]=ClassicBPE.from_json(json.loads(path.read_text()))
  full=f'{y}_fullpool.json';models[full]=ClassicBPE.from_json(json.loads((MODELS/full).read_text()))
  valid=np.ones(len(words),dtype=bool)
  for model in models.values():
   valid &= np.array([all(ch in model.alphabet for ch in w) for w in words])
  umask=valid & unseen
  coverage=float((freq*lengths*umask).sum()/max(1,(freq*lengths*unseen).sum()))
  assert coverage>.97,coverage
  prices={};jacs={}
  for name,model in models.items():
   c=np.array([model.encode_len(w) if valid[i] else 0 for i,w in enumerate(words)],dtype=np.int32)
   prices[name]=float(100*(c*freq*umask).sum()/(freq*lengths*umask).sum())
  for row in rr:
   a,b=row['candidates'];g=prices[a['file']];s=prices[b['file']];ch=prices[row['selected_file']]
   row['outcome']={'greedy_E':g,'second_E':s,'selected_E':ch,'regret':g-min(g,s),'gain':g-ch}
   va={''.join(p) for p in json.loads((MODELS/a['file']).read_text())['merges']}
   vb={''.join(p) for p in json.loads((MODELS/b['file']).read_text())['merges']}
   row['vocab_jaccard']=len(va&vb)/len(va|vb)
  x=np.array([r['outcome']['gain'] for r in rr])
  bs=x[rng.integers(0,len(x),size=(10000,len(x)))].mean(axis=1)
  stats={'first_pairs':dict(collections.Counter(''.join(r['top_pairs'][0][0]) for r in rr)),
         'flag_count':sum(r['flag'] for r in rr),
         'material_error_count':sum(r['outcome']['regret']>=.25 for r in rr),
         'greedy_mean_E':float(np.mean([r['outcome']['greedy_E'] for r in rr])),
         'selected_mean_E':float(np.mean([r['outcome']['selected_E'] for r in rr])),
         'fullpool_E':prices[full],
         'mean_gain':float(x.mean()),'gain_ci95':np.quantile(bs,[.025,.975]).tolist(),
         'mean_jaccard':float(np.mean([r['vocab_jaccard'] for r in rr])),
         'unseen_coverage':coverage,
         'rows':rr}
  stats['temporal_gate']=len(stats['first_pairs'])>1 and stats['material_error_count']>=4 and stats['mean_gain']>=.25 and stats['gain_ci95'][0]>0
  out[str(y)]=stats
  print('TEMP_SCORED',y, {k:v for k,v in stats.items() if k!='rows'},flush=True)
 RESULT.write_text(json.dumps({'precommit_sha':(P/'TEMPORAL_PRECOMMIT.sha256').read_text().split()[0],
    'results':out,'both_temporal_gates_pass':all(x['temporal_gate'] for x in out.values()),
    'retrospective_only':True},indent=2,ensure_ascii=False))

if __name__=='__main__':
 cmd=sys.argv[1]
 if cmd=='prepare':get_data()
 elif cmd=='train':run_train(int(sys.argv[2]),int(sys.argv[3]))
 elif cmd=='fullpool':full()
 elif cmd=='score':score()
 else:raise ValueError(cmd)
