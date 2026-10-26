#!/usr/bin/env python3
"""Phase-4 F6 evaluation: only runs after training manifest and all model hashes are frozen."""
from __future__ import annotations
import argparse,csv,hashlib,json,sys,collections
from pathlib import Path
import numpy as np
import pandas as pd
import sentencepiece as spm

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Phase3'/'scripts'))
from run_phase3 import load_heldout,digest

EXPECTED_ARCHIVE='375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5'
PRECOMMIT_SHA='1fb49e638a51cedc7f8157743d2c2d90ba65003f4d049990cf8d23c30f00714e'
PHASE3_PAGES_SHA='e6adc3271c666b56655e676b325c1f747618f7d88ead5749bbf85ffe651a8cbb'
SEEDS=(441,442,443,444,445)
CONDITIONS=('filtered','support_control')
ARMS=('PP','WW')
PERIODS=('1888','1889-94')
VIEWS=('en','xh')
NBOOT=5000

def read_csv(p):
 with open(p,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def write_csv(p,rows,cols=None):
 if not rows:raise ValueError('empty')
 with open(p,'w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=cols or list(rows[0]),extrasaction='ignore');w.writeheader();w.writerows(rows)

def get_models(out):
 lock=json.loads((out/'TRAINING_LOCK.json').read_text())
 assert lock['training_complete'] and lock['model_count']==20 and lock['test_content_read'] is False
 assert lock['precommit_sha256']==PRECOMMIT_SHA
 assert digest(out/'model_training_manifest.csv')==lock['model_manifest_sha256']
 assert digest(out/'support_manifest.csv')==lock['support_manifest_sha256']
 manifest=read_csv(out/'model_training_manifest.csv')
 seen=set(); models=[]
 for r in manifest:
  cond,arm,seed=r['condition'],r['arm'],int(r['seed'])
  assert cond in CONDITIONS and arm in ARMS and seed in SEEDS
  assert (cond,arm,seed) not in seen;seen.add((cond,arm,seed))
  path=out/'models'/r['model_name']
  assert digest(str(path)+'.model')==r['model_sha256']
  assert digest(str(path)+'.vocab')==r['vocab_sha256']
  assert 499970<=int(r['actual_alpha_chars'])<=500000
  assert int(r['vocab_size'])==2000
  assert int(r['training_only_repeated_shingles_found'])==0 if cond=='filtered' else True
  models.append((cond,arm,seed,path.with_suffix('.model')))
 assert len(seen)==20
 return models,lock

def heldout_assertions(root,pages):
 assert digest(root/'Phase3'/'outputs'/'heldout_pageview_manifest.csv')==PHASE3_PAGES_SHA
 earlier=read_csv(root/'Phase3'/'outputs'/'heldout_pageview_manifest.csv')
 refs={(r['filename'],r['view']):r for r in earlier}
 assert len(refs)==len(pages)==2493
 for r in pages:
  k=(r['filename'],r['view']);a=refs[k]
  for f in ('period','text_sha256','issue','view','filename'):
   assert str(a[f])==str(r[f]),(f,k)
  for f in ('year','chars','words','total_view_words'):
   assert int(a[f])==int(r[f]),(f,k)

def score(pages,models):
 out=[]
 for ix,(cond,arm,seed,path) in enumerate(models,1):
  sp=spm.SentencePieceProcessor(model_file=str(path))
  assert sp.vocab_size()==2000
  for p in pages:
   ids=sp.encode(p['text'],out_type=int)
   out.append(dict(condition=cond,arm=arm,seed=seed,filename=p['filename'],issue=p['issue'],year=p['year'],
        period=p['period'],view=p['view'],chars=p['chars'],text_sha256=p['text_sha256'],
        pieces=len(ids),unknown=sum(int(x==sp.unk_id()) for x in ids)))
  print('SCORE',ix,'/',len(models),cond,arm,seed,flush=True)
 assert len(out)==20*len(pages)
 return pd.DataFrame(out)

def bootstrap(df):
 out=[];seedrows=[];metarows=[]
 for period in PERIODS:
  d=df[df.period==period]
  issues=sorted(d.issue.unique());n=len(issues)
  assert n== (52 if period=='1888' else 306)
  rng=np.random.default_rng(343401+(period!='1888'))
  weights=rng.multinomial(n,[1/n]*n,size=NBOOT).astype(float)
  if period=='1889-94':
   years=list(range(1889,1895))
   yearrng=np.random.default_rng(343501)
   yearweights=yearrng.multinomial(6,[1/6]*6,size=NBOOT).astype(float)
  metarows.append(dict(period=period,issues=n,draws=NBOOT,random_seed=343401+(period!='1888'),
      year_cluster_seed=343501 if period=='1889-94' else ''))
  for view in VIEWS:
   q=d[d.view==view]
   cb=q[(q.condition=='filtered')&(q.arm=='PP')&(q.seed==441)].groupby('issue').chars.sum()
   ch=np.array([float(cb.get(i,0.)) for i in issues]);assert ch.sum()>0
   den=weights@ch;assert (den>0).all()
   P=np.zeros((2,2,5,n))
   for j,cond in enumerate(CONDITIONS):
    for k,arm in enumerate(ARMS):
     for m,s in enumerate(SEEDS):
      sub=q[(q.condition==cond)&(q.arm==arm)&(q.seed==s)]
      cc=sub.groupby('issue').chars.sum()
      assert np.allclose([cc.get(i,0) for i in issues],ch)
      pc=sub.groupby('issue').pieces.sum()
      P[j,k,m]=np.array([float(pc.get(i,0)) for i in issues])
   observed=100*P.sum(axis=-1)/ch.sum()   # condition,arm,seed
   sampled=(weights@P.reshape(20,n).T).reshape(NBOOT,2,2,5)
   Es=100*sampled/den[:,None,None,None]
   effects=observed[:,0,:]-observed[:,1,:]
   beffects=Es[:,:,0,:]-Es[:,:,1,:]
   contrasts={
     'filtered_PP_minus_WW':effects[0],
     'support_control_PP_minus_WW':effects[1],
     'filtered_minus_support_control':effects[0]-effects[1],
     'filtered_PP_minus_control_PP':observed[0,0]-observed[1,0],
     'filtered_WW_minus_control_WW':observed[0,1]-observed[1,1]
   }
   boots={
     'filtered_PP_minus_WW':beffects[:,0,:],
     'support_control_PP_minus_WW':beffects[:,1,:],
     'filtered_minus_support_control':beffects[:,0,:]-beffects[:,1,:],
     'filtered_PP_minus_control_PP':Es[:,0,0,:]-Es[:,1,0,:],
     'filtered_WW_minus_control_WW':Es[:,0,1,:]-Es[:,1,1,:]
   }
   if period=='1889-94':
    yrs=[int(i.split('_')[1][:4]) for i in issues]
    py=np.zeros((2,2,5,6));yc=np.zeros(6)
    for idx,y in enumerate(yrs):
     yy=years.index(y);py[:,:,:,yy]+=P[:,:,:,idx];yc[yy]+=ch[idx]
    yp=(yearweights@py.reshape(20,6).T).reshape(NBOOT,2,2,5)
    yc_boot=yearweights@yc;assert (yc_boot>0).all()
    Y=100*yp/yc_boot[:,None,None,None]
    yrs_boot={
      'filtered_PP_minus_WW':Y[:,0,0,:]-Y[:,0,1,:],
      'support_control_PP_minus_WW':Y[:,1,0,:]-Y[:,1,1,:],
      'filtered_minus_support_control':(Y[:,0,0,:]-Y[:,0,1,:])-(Y[:,1,0,:]-Y[:,1,1,:]),
      'filtered_PP_minus_control_PP':Y[:,0,0,:]-Y[:,1,0,:],
      'filtered_WW_minus_control_WW':Y[:,0,1,:]-Y[:,1,1,:]
    }
   else:yrs_boot={}
   for key,x in contrasts.items():
    vals=np.asarray(x,dtype=float);bv=np.asarray(boots[key]).mean(axis=-1)
    lo,hi=np.quantile(bv,[.025,.975]);r=dict(period=period,view=view,contrast=key,mean=float(vals.mean()),
       ci_low=float(lo),ci_high=float(hi),seeds=5,seed_positive=int((vals>0).sum()),
       seed_negative=int((vals<0).sum()),seed_min=float(vals.min()),seed_max=float(vals.max()),
       issues=n,chars=float(ch.sum()),bootstrap_draws=NBOOT)
    if key in yrs_boot:
     ymean=np.asarray(yrs_boot[key]).mean(axis=-1)
     ylo,yhi=np.quantile(ymean,[.025,.975]);r.update(year_ci_low=float(ylo),year_ci_high=float(yhi))
    else:r.update(year_ci_low='',year_ci_high='')
    out.append(r)
    for s,v in zip(SEEDS,vals):seedrows.append(dict(period=period,view=view,contrast=key,seed=s,effect=float(v)))
   for j,cond in enumerate(CONDITIONS):
    for k,arm in enumerate(ARMS):
     for m,s in enumerate(SEEDS):
      metarows.append(dict(period=period,view=view,condition=cond,arm=arm,seed=s,
        pieces_per_100_chars=float(observed[j,k,m]),chars=float(ch.sum()),issues=n))
 return out,seedrows,metarows

def train_test_overlap(root,archive,pages):
 sys.path.insert(0,str(ROOT/'Phase1'/'scripts'))
 from audit_phase1 import parse,enumerate_units,K
 train,_,_,_=parse(archive)
 train_units,_=enumerate_units(train)
 from collections import defaultdict
 byview=defaultdict(set)
 for u in train_units:
  w=u['words'];byview[u['view']].update(tuple(w[i:i+K]) for i in range(len(w)-K+1))
 rows=[]
 for p in pages:
  w=p['text'].split();den=len(w)-K+1
  hits=sum(tuple(w[i:i+K]) in byview[p['view']] for i in range(den))
  rows.append(dict(filename=p['filename'],issue=p['issue'],view=p['view'],period=p['period'],
     shingle_types_considered=den,train_overlap_13grams=hits,has_train_overlap=int(hits>0)))
 return rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--archive',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
 out=Path(args.output)
 assert digest(args.archive)==EXPECTED_ARCHIVE
 assert digest(ROOT/'Phase4'/'PRECOMMIT.md')==PRECOMMIT_SHA
 models,lock=get_models(out)
 pages=load_heldout(args.archive)
 heldout_assertions(ROOT,pages)
 df=score(pages,models)
 df.to_csv(out/'heldout_scores.csv',index=False)
 summ,seedrows,meta=bootstrap(df)
 write_csv(out/'f6_contrasts.csv',summ)
 write_csv(out/'f6_seed_effects.csv',seedrows)
 write_csv(out/'f6_support_metrics.csv',meta,list(meta[0]))
 overlap=train_test_overlap(ROOT,args.archive,pages)
 write_csv(out/'train_test_overlap_diagnostic.csv',overlap)
 a=next(r for r in summ if r['period']=='1888' and r['view']=='en' and r['contrast']=='filtered_PP_minus_WW')
 b=next(r for r in summ if r['period']=='1889-94' and r['view']=='en' and r['contrast']=='filtered_PP_minus_WW')
 accepted=a['mean']>0.30 and a['ci_low']>0 and b['mean']>0
 prior=read_csv(ROOT/'Phase3'/'outputs'/'factorial_contrasts.csv')
 original=[]
 for period in PERIODS:
  for v in VIEWS:
   matches=[r for r in prior if r['family']=='bpe' and r['period']==period and r['view']==v and r['contrast']=='WW_minus_PP']
   assert len(matches)==1
   r=matches[0];original.append(dict(period=period,view=v,
      original_PP_minus_WW=-float(r['mean']),original_ci_low=-float(r['ci_high']),original_ci_high=-float(r['ci_low'])))
 write_csv(out/'phase3_original_reference.csv',original)
 summary={
   'phase':'4','frozen_F6_status':'SUPPORTED' if accepted else 'NOT_SUPPORTED',
   'criterion':'1888 English filtered PP-WW >+0.30, lower CI >0 and 1889-94 English filtered PP-WW >0',
   'effect_1888_english':a,'effect_1889_94_english':b,
   'model_count':len(models),'model_view_scores':len(df),'page_views_scored':len(pages),
   'unknown_token_total':int(df.unknown.sum()),'unseen_pageview_count':len(pages),
   'train_test_overlap_13gram_fraction':{period:{v:sum(x['train_overlap_13grams'] for x in overlap if x['period']==period and x['view']==v)/sum(x['shingle_types_considered'] for x in overlap if x['period']==period and x['view']==v) for v in VIEWS} for period in PERIODS},
   'archive_sha256':EXPECTED_ARCHIVE,'precommit_sha256':PRECOMMIT_SHA,'training_lock_sha256':digest(out/'TRAINING_LOCK.json'),
   'phase3_heldout_manifest_sha256':PHASE3_PAGES_SHA,
   'interpretation_warning':'Conditional follow-up; prior phases already inspected the test years; archive view labels are not independently verified language identities.'}
 (out/'PHASE4_DECISION.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
 print('F6',summary['frozen_F6_status'],'1888',a['mean'],a['ci_low'],a['ci_high'],
       'later',b['mean'],b['ci_low'],b['ci_high'],'unknown',summary['unknown_token_total'],flush=True)

if __name__=='__main__':main()
