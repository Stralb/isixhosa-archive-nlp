#!/usr/bin/env python3
"""Independent source reconstruction, independent deterministic sampling and bootstrap.
Does not import Phase-4 train or evaluation programs. Checks all model inputs and outputs.
"""
import argparse,csv,hashlib,json,re,zipfile,collections,math
from pathlib import Path
import numpy as np
import pandas as pd
import sentencepiece as spm

T=re.compile(r'[a-z]+');NAME=re.compile(r'IMVO_(\d{4})-(\d{2})-(\d{2})_p\d{3}\.txt$')
K=13;WIN=32;SEEDS=[441,442,443,444,445]
CONDITIONS=['filtered','support_control']; ARMS=['PP','WW']; VIEWS=['en','xh']

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for ch in iter(lambda:f.read(1024*1024),b''):h.update(ch)
 return h.hexdigest()

def readcsv(path):
 with open(path,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def rebuild(archive):
 train={};test={}
 with zipfile.ZipFile(archive) as z:
  for member in z.namelist():
   parts=member.split('/')
   if len(parts)<2 or parts[-2] not in ('EnTxt','XhTxt'):continue
   fname=parts[-1];mm=NAME.fullmatch(fname)
   if not mm:continue
   y,m,d=(int(x) for x in mm.groups())
   view='en' if parts[-2]=='EnTxt' else 'xh'
   if (y==1885 or (y==1887 and m<=9)):
    k=train.setdefault(fname,dict(views={},issue=f'IMVO_{y:04d}-{m:02d}-{d:02d}',file=fname))
    k['views'][view]=T.findall(z.read(member).decode('utf-8-sig',errors='replace').lower())
   elif 1888<=y<=1894:
    ww=T.findall(z.read(member).decode('utf-8-sig',errors='replace').lower())
    if len(ww)<40:continue
    text=' '.join(ww[:128]);test[(fname,view)]={'filename':fname,'view':view,'text':text,
      'chars':sum(map(len,ww[:128])),'words':len(ww[:128]),'total_view_words':len(ww),
      'text_sha256':hashlib.sha256(text.encode()).hexdigest(),
      'issue':f'IMVO_{y:04d}-{m:02d}-{d:02d}','year':y,
      'period':'1888' if y==1888 else '1889-94'}
 assert len(train)==342 and len(test)==2493
 units=[]
 for fname in sorted(train):
  a=train[fname];assert set(a['views'])=={'en','xh'}
  for view in ('xh','en'):
   ww=a['views'][view]
   if len(ww)>=16:units.append(dict(file=fname,issue=a['issue'],view=view,words=ww,nwords=len(ww)))
 assert len(units)==595
 return units,test

def make_support(units):
 first={};banned=set()
 for u in units:
  w=u['words'];issue=u['issue']
  for i in range(len(w)-K+1):
   g=tuple(w[i:i+K]);owner=first.setdefault(g,issue)
   if owner!=issue:banned.add(g)
 assert len(banned)==16691
 filtered=[];control=[]
 for u in units:
  w=u['words'];n=len(w)
  flags=np.fromiter((tuple(w[i:i+K]) in banned for i in range(n-K+1)),dtype=np.int8,count=n-K+1)
  count=max(n-WIN+1,1)
  if n>=WIN:
   pref=np.r_[0,flags.cumsum()]
   clean=np.flatnonzero(pref[np.arange(count)+WIN-K+1]-pref[np.arange(count)]==0)
  else:clean=np.array([0],dtype=int) if flags.sum()==0 else np.array([],dtype=int)
  if not len(clean):continue
  filtered.append({**u,'starts':clean});control.append({**u,'starts':np.arange(count,dtype=int)})
 assert len(filtered)==len(control)==533
 return filtered,control,banned,first

def re_draw(pool,arm,seed):
 bypage=collections.defaultdict(lambda:{'en':0,'xh':0})
 for u in pool:bypage[u['file']][u['view']]=u['nwords']
 vcounts=list(bypage.values())
 p_en=sum(v['en']/(v['en']+v['xh']) for v in vcounts)/len(vcounts) if arm[0]=='P' else sum(v['en'] for v in vcounts)/sum(v['en']+v['xh'] for v in vcounts)
 vids={v:[i for i,u in enumerate(pool) if u['view']==v] for v in ('en','xh')}
 probs={}
 for v in vids:
  if arm[1]=='P':
   w=np.array([pool[i]['nwords']/sum(bypage[pool[i]['file']].values()) for i in vids[v]],dtype=float)
  else:w=np.array([pool[i]['nwords'] for i in vids[v]],dtype=float)
  probs[v]=w/w.sum()
 rng=np.random.default_rng(int(seed)*1000+{'PP':11,'WW':14}[arm]);chars=0;en_chars=0;lines=[];windows=set();issues=set()
 while chars<500000-30:
  view='en' if rng.random()<p_en else 'xh'
  unit=pool[vids[view][int(rng.choice(len(vids[view]),p=probs[view]))]]
  start=int(unit['starts'][int(rng.integers(len(unit['starts'])))])
  block=unit['words'][start:start+WIN]
  remaining=500000-chars;keep=[];add=0
  for word in block:
   if add+len(word)>remaining:break
   keep.append(word);add+=len(word)
  if not keep:break
  line=' '.join(keep);lines.append(line);chars+=add;en_chars+=add*(view=='en')
  windows.add(hashlib.sha256(line.encode()).digest());issues.add(unit['issue'])
 return lines,dict(sample_hash_sha256=hashlib.sha256(('\n'.join(lines)+'\n').encode()).hexdigest(),
  actual_alpha_chars=chars,en_alpha_chars=en_chars,en_alpha_share=en_chars/chars,
  lines=len(lines),unique_lines=len(windows),source_issues_represented=len(issues))

def check_bootstrap(scores,reported):
 detail=[];maxdiff=0.
 for period in ['1888','1889-94']:
  issues=sorted(scores[scores.period==period].issue.unique());n=len(issues)
  weights=np.random.default_rng(343401+(period!='1888')).multinomial(n,[1/n]*n,size=5000).astype(float)
  if period=='1889-94':
   yearweights=np.random.default_rng(343501).multinomial(6,[1/6]*6,size=5000).astype(float)
  for view in VIEWS:
   sub=scores[(scores.period==period)&(scores.view==view)]
   chars_by_issue=sub[(sub.condition=='filtered')&(sub.arm=='PP')&(sub.seed==441)].groupby('issue').chars.sum()
   chars=np.array([float(chars_by_issue.get(i,0)) for i in issues]);assert chars.sum()>0
   pieces=np.zeros((2,2,5,n),dtype=float)
   for j,cond in enumerate(CONDITIONS):
    for k,arm in enumerate(ARMS):
     for sidx,seed in enumerate(SEEDS):
      r=sub[(sub.condition==cond)&(sub.arm==arm)&(sub.seed==seed)]
      counts=r.groupby('issue').pieces.sum()
      pieces[j,k,sidx]=[counts.get(i,0) for i in issues]
      counts_chars=r.groupby('issue').chars.sum()
      assert np.array_equal(chars,np.array([counts_chars.get(i,0) for i in issues]))
   base=100*pieces.sum(axis=3)/chars.sum()
   sample=100*(weights @ pieces.reshape(20,n).T).reshape(5000,2,2,5)/(weights@chars)[:,None,None,None]
   contrasts={
    'filtered_PP_minus_WW':(base[0,0]-base[0,1],sample[:,0,0]-sample[:,0,1]),
    'support_control_PP_minus_WW':(base[1,0]-base[1,1],sample[:,1,0]-sample[:,1,1]),
    'filtered_minus_support_control':((base[0,0]-base[0,1])-(base[1,0]-base[1,1]),
       (sample[:,0,0]-sample[:,0,1])-(sample[:,1,0]-sample[:,1,1])),
    'filtered_PP_minus_control_PP':(base[0,0]-base[1,0],sample[:,0,0]-sample[:,1,0]),
    'filtered_WW_minus_control_WW':(base[0,1]-base[1,1],sample[:,0,1]-sample[:,1,1])}
   if period=='1889-94':
    yearpieces=np.zeros((2,2,5,6));yc=np.zeros(6)
    for j,issue in enumerate(issues):
     iy=int(issue[5:9])-1889;yearpieces[:,:,:,iy]+=pieces[:,:,:,j];yc[iy]+=chars[j]
    ysample=100*(yearweights @ yearpieces.reshape(20,6).T).reshape(5000,2,2,5)/(yearweights@yc)[:,None,None,None]
    yc_contrasts={'filtered_PP_minus_WW':ysample[:,0,0]-ysample[:,0,1],
       'support_control_PP_minus_WW':ysample[:,1,0]-ysample[:,1,1],
       'filtered_minus_support_control':(ysample[:,0,0]-ysample[:,0,1])-(ysample[:,1,0]-ysample[:,1,1]),
       'filtered_PP_minus_control_PP':ysample[:,0,0]-ysample[:,1,0],
       'filtered_WW_minus_control_WW':ysample[:,0,1]-ysample[:,1,1]}
   for contrast,(val,boot) in contrasts.items():
    rr=reported[(reported.period==period)&(reported.view==view)&(reported.contrast==contrast)]
    assert len(rr)==1;row=rr.iloc[0]
    lo,hi=np.quantile(boot.mean(axis=1),[.025,.975]);theory=[val.mean(),lo,hi]
    output=[float(row['mean']),float(row['ci_low']),float(row['ci_high'])]
    md=max(abs(float(a)-float(b)) for a,b in zip(theory,output));maxdiff=max(maxdiff,md)
    assert md<2e-11,(period,view,contrast,md)
    if period=='1889-94':
     ylo,yhi=np.quantile(yc_contrasts[contrast].mean(axis=1),[.025,.975])
     assert max(abs(ylo-float(row['year_ci_low'])),abs(yhi-float(row['year_ci_high'])))<2e-11
    detail.append(dict(period=period,view=view,contrast=contrast,max_difference=md))
 return {'contrasts_recomputed':len(detail),'max_absolute_difference':maxdiff}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--archive',required=True);ap.add_argument('--root',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
 root=Path(args.root);out=Path(args.output)
 assert sha(args.archive)=='375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5'
 assert sha(root/'Phase1'/'FROZEN_PROTOCOL.md')=='bc443d883fa1c2b8eb09ced5f87d8f65b352d55512ef2c594b1b23e64d496bf4'
 assert sha(root/'Phase4'/'PRECOMMIT.md')=='1fb49e638a51cedc7f8157743d2c2d90ba65003f4d049990cf8d23c30f00714e'
 assert sha(root/'Phase3'/'outputs'/'heldout_pageview_manifest.csv')=='e6adc3271c666b56655e676b325c1f747618f7d88ead5749bbf85ffe651a8cbb'
 units,heldout=rebuild(args.archive)
 filt,ctrl,banned,first=make_support(units)
 keys={(u['file'],u['view']) for u in filt};assert len(keys)==533
 support=readcsv(out/'support_manifest.csv');assert len(support)==533
 fdict={(u['file'],u['view']):u for u in filt}
 cdict={(u['file'],u['view']):u for u in ctrl}
 for r in support:
  key=(r['filename'],r['view']);u=fdict[key];c=cdict[key]
  assert key==(u['file'],u['view'])==(c['file'],c['view'])
  assert int(r['clean_windows'])==len(u['starts']) and int(r['all_windows'])==len(c['starts'])
 lock=json.loads((out/'TRAINING_LOCK.json').read_text())
 assert sha(out/'model_training_manifest.csv')==lock['model_manifest_sha256']
 models=readcsv(out/'model_training_manifest.csv');assert len(models)==20
 max_sample_diff=0.;nline=0;dirty_control=0
 for j,r in enumerate(models,1):
  cond,arm,seed=r['condition'],r['arm'],int(r['seed']);pool=filt if cond=='filtered' else ctrl
  lines,meta=re_draw(pool,arm,seed)
  for k in ('sample_hash_sha256','actual_alpha_chars','en_alpha_chars','lines','unique_lines','source_issues_represented'):
   assert str(meta[k])==str(r[k]),(r['model_name'],k,meta[k],r[k])
  assert abs(meta['en_alpha_share']-float(r['en_alpha_share']))<1e-14
  repeats=sum(tuple(parts[i:i+K]) in banned for line in lines for parts in [line.split()] for i in range(len(parts)-K+1))
  assert int(r['training_only_repeated_shingles_found'])==repeats
  if cond=='filtered':assert repeats==0
  else:dirty_control+=repeats
  path=out/'models'/r['model_name'];assert sha(str(path)+'.model')==r['model_sha256']
  assert sha(str(path)+'.vocab')==r['vocab_sha256']
  assert spm.SentencePieceProcessor(model_file=str(path)+'.model').vocab_size()==2000
  nline+=len(lines)
  print('VERIFIED TRAIN INPUT',j,'of 20',r['model_name'],flush=True)
 assert dirty_control>0
 refs=readcsv(root/'Phase3'/'outputs'/'heldout_pageview_manifest.csv');assert len(refs)==2493
 for r in refs:
  a=heldout[(r['filename'],r['view'])]
  assert r['text_sha256']==a['text_sha256'] and int(r['chars'])==a['chars']
 scores=pd.read_csv(out/'heldout_scores.csv');assert len(scores)==20*2493
 pmap={r['model_name']:spm.SentencePieceProcessor(model_file=str(out/'models'/r['model_name'])+'.model') for r in models}
 mismatches=0;total_unk=0
 for idx,r in enumerate(scores.itertuples(index=False),1):
  model=pmap[f'bpe_{r.condition}_{r.arm}_{r.seed}'];doc=heldout[(r.filename,r.view)]
  toks=model.encode(doc['text'],out_type=int)
  mismatches+=len(toks)!=r.pieces or doc['text_sha256']!=r.text_sha256 or doc['chars']!=r.chars
  total_unk+=sum(x==model.unk_id() for x in toks)
  if idx%10000==0:print('VERIFIED EVALUATION',idx,'of',len(scores),flush=True)
 assert mismatches==0 and total_unk==0
 # Independent source-only construction of train/test shingle-overlap sensitivity.
 train_grams={v:set() for v in VIEWS}
 for u in units:
  w=u['words']; train_grams[u['view']].update(tuple(w[i:i+K]) for i in range(len(w)-K+1))
 overlap=readcsv(out/'train_test_overlap_diagnostic.csv')
 assert len(overlap)==2493
 for r in overlap:
  doc=heldout[(r['filename'],r['view'])]; w=doc['text'].split()
  overlaps=sum(tuple(w[i:i+K]) in train_grams[r['view']] for i in range(len(w)-K+1))
  assert overlaps==int(r['train_overlap_13grams'])
  assert len(w)-K+1==int(r['shingle_types_considered'])
  assert int(overlaps>0)==int(r['has_train_overlap'])
 contrasts=pd.read_csv(out/'f6_contrasts.csv');bootinfo=check_bootstrap(scores,contrasts)
 q=next(r for r in readcsv(out/'f6_contrasts.csv') if r['period']=='1888' and r['view']=='en' and r['contrast']=='filtered_PP_minus_WW')
 supported=float(q['mean'])>0.3 and float(q['ci_low'])>0 and float(next(r['mean'] for r in readcsv(out/'f6_contrasts.csv') if r['period']=='1889-94' and r['view']=='en' and r['contrast']=='filtered_PP_minus_WW'))>0
 decision=json.loads((out/'PHASE4_DECISION.json').read_text());assert supported==(decision['frozen_F6_status']=='SUPPORTED')
 cert={'status':'PASS','phase':'4','verified_train_models':len(models),'verified_training_sample_lines':nline,
   'training_samples_rebuilt_independently':True,'same_filtered_control_page_view_support':True,
   'filtered_repeated_shingle_violations':0,'control_repeated_shingle_occurrences':dirty_control,
   'cross_issue_repeated_shingle_types':len(banned),'heldout_pageviews_rebuilt_from_archive':len(heldout),
   'verified_heldout_model_scores':len(scores),'tokenization_score_mismatches':mismatches,
   'unknown_tokens':total_unk,'train_test_shingle_rows_recomputed':len(overlap),**bootinfo,'F6_status':decision['frozen_F6_status'],
   'phase4_precommit_sha256':sha(root/'Phase4'/'PRECOMMIT.md'),
   'archive_sha256':sha(args.archive),'model_training_manifest_sha256':sha(out/'model_training_manifest.csv'),
   'source_archive_required_for_full_rerun':True}
 (out/'PHASE4_INDEPENDENT_VERIFICATION.json').write_text(json.dumps(cert,indent=2,sort_keys=True)+'\n')
 print('INDEPENDENT VERIFY',json.dumps(cert,sort_keys=True),flush=True)

if __name__=='__main__':main()
