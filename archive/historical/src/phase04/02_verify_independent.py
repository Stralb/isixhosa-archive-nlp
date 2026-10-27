#!/usr/bin/env python3
"""Independent Phase4 exhaustive audit; direct page summation, no import of scoring code."""
import argparse,collections,hashlib,json,re,zipfile
from pathlib import Path
import numpy as np,pandas as pd
CAP=.02;THRESH=.0002
PH3_EXPECT={'dev':'47551f5db490029f563b77833f4d0eaabd0ea4ce5dbdd43771b821348c95a809','test':'67b2dec56e8ad7dfc87896aa0742604afac25a60b83df02c9a84de92cacc68c9'}
ZIP_HASH='375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5'
RX=re.compile(r'[^\W\d_]+',re.UNICODE)

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as fh:
  for block in iter(lambda:fh.read(1<<20),b''):h.update(block)
 return h.hexdigest()

def blinded_order(obs,yr,groups,method,tag):
 na=int((yr==0).sum());nb=int((yr==1).sum())
 pool=np.flatnonzero(~obs)
 d=hashlib.sha256('|'.join(map(str,('P4-RECOVERY',20261024)+tag+(method,))).encode()).digest()
 pp=np.random.default_rng(int.from_bytes(d[:8],'big')).permutation(pool)
 if method=='RANDOM':return pp
 large=np.array([1/na if yr[i]==0 else 1/nb for i in pp])
 greedy=pp[np.argsort(-large,kind='stable')]
 if method=='WIDTH_GREEDY':return greedy
 assert method=='CELL_REPAIR'
 chosen=[]
 for year in (0,1):
  for c in range(4):
   selected=np.flatnonzero((yr==year)&(groups==c))
   if selected.size and not np.any(obs[selected]):
    matched=np.intersect1d(pp,selected,assume_unique=True)
    # Prefer first element in randomized order, not sorted by row ID
    take=next(int(i) for i in pp if i in set(map(int,matched)))
    chosen.append(take)
 chosen_set=set(chosen)
 return np.array(chosen+[int(i) for i in greedy if int(i) not in chosen_set],dtype=int)

def verify_phase(phase3,phase4,split):
 inp=Path(phase3)/'outputs'/split/'benchmark.npz'
 assert digest(inp)==PH3_EXPECT[split]
 with np.load(inp,allow_pickle=False) as old: z={k:old[k] for k in old.files}
 with np.load(Path(phase4)/'outputs'/split/'bounds_and_audits.npz',allow_pickle=False) as saved:
  saved={k:saved[k] for k in saved.files}
 orders=pd.read_csv(Path(phase4)/'outputs'/split/'audit_orders_hashes.csv')
 report=pd.read_csv(Path(phase4)/'outputs'/split/'audit_replicate_results.csv')
 scen=list(map(str,z['scenarios']));pol=list(map(str,saved['policies']));bud=saved['budgets'];R=z['masks'].shape[1]
 assert pol==['RANDOM','WIDTH_GREEDY','CELL_REPAIR'] and len(bud)==8
 scores=[];pages=z['page_totals'];year=z['period'];cell=z['cells'];NF=80
 NN=np.bincount(year,minlength=2)
 Y=np.stack([np.minimum(z['base_counts']/pages[:,None],CAP),np.minimum(z['injected_counts']/pages[:,None],CAP)])
 truth=np.array([YY[year==1].mean(axis=0)-YY[year==0].mean(axis=0) for YY in Y])
 assert np.allclose(truth,saved['truth'],atol=1e-14)
 index_records=report.set_index(['scenario','replicate','world','policy','budget'])
 idx_ord=orders.set_index(['scenario','replicate','world','policy'])
 maximal_delta=0.
 total_vectors=0
 for si,s in enumerate(scen):
  for rep in range(R):
   for world in range(2):
    YY=Y[world];mask=z['masks'][si,rep,world]
    for pi,policy in enumerate(pol):
     order=blinded_order(mask,year,cell,policy,(split,s,rep,world))
     tag=(s,rep,world,policy)
     assert hashlib.sha256(order.astype('<i4').tobytes()).hexdigest()==idx_ord.loc[tag,'order_sha256']
     for qi,q in enumerate(bud):
      k=int(np.floor(q*len(order))) if q<1 else len(order)
      newseen=mask.copy();newseen[order[:k]]=True
      bottom=[];top=[]
      for t in [0,1]:
       one=YY[newseen & (year==t)].sum(axis=0)
       holes=int(np.count_nonzero(~newseen & (year==t)))
       bottom.append(one/NN[t]);top.append((one+CAP*holes)/NN[t])
      low=bottom[1]-top[0];high=top[1]-bottom[0]
      expected=np.column_stack([low,high])
      actual=saved['lower_upper'][si,rep,world,pi,qi]
      delta=np.max(np.abs(expected-actual))
      maximal_delta=max(maximal_delta,float(delta))
      assert delta<4e-13,(tag,q,delta)
      assert np.all((low<truth[world]+1e-12)&(high>truth[world]-1e-12))
      assert k==int(saved['audited_counts'][si,rep,world,pi,qi])
      e=int(sum(bool(np.any((year==t)&(cell==c))) and not bool(np.any(newseen&(year==t)&(cell==c))) for t in (0,1) for c in range(4)))
      assert e==saved['empty_cells_remaining'][si,rep,world,pi,qi]
      dec=np.logical_or(low>THRESH,high<-THRESH)
      row=index_records.loc[(s,rep,['NATURAL','INJECTED'][world],policy,q)]
      assert np.isclose(row['bound_width_per_10k'],(high-low).mean()*10000,atol=1e-10)
      assert np.isclose(row['practical_sign_coverage'],dec.mean(),atol=1e-12)
      assert int(row['practical_sign_count'])==dec.sum()
      assert int(row['empty_cells_after'])==e
      assert int(row['wrong_calls'])==0
      total_vectors+=1
  print('verified',split,s,flush=True)
 assert total_vectors==len(report)
 return {'split':split,'fully_recomputed_bound_vectors':total_vectors,
         'per_feature_bounds_verified':total_vectors*NF,'max_numerical_delta':maximal_delta,
         'saved_output_rows':len(report),'status':'PASS'}

def raw_text_check(phase3,source,limit=32):
 assert digest(source)==ZIP_HASH
 frame=pd.read_csv(Path(phase3)/'inputs'/'frame.csv')
 fset=pd.read_csv(Path(phase3)/'inputs'/'feature_manifest.csv',keep_default_na=False)
 vocab=list(fset.word_form)
 with np.load(Path(phase3)/'inputs'/'counts.npz') as zz:cnt=zz['counts']
 assert len(frame)==len(cnt)
 rng=np.random.default_rng(20261024)
 selected=np.concatenate([np.sort(rng.choice(np.flatnonzero(frame['split']==which),size=limit//2,replace=False)) for which in ('dev','test')])
 with zipfile.ZipFile(source) as z:
  for i in selected:
   page=z.read(frame.loc[i,'zip_entry'])
   toks=RX.findall(page.decode('utf-8','replace').casefold())
   assert len(toks)==int(frame.loc[i,'n_tokens'])
   cc=collections.Counter(toks)
   assert np.array_equal(cnt[i],np.asarray([cc[w] for w in vocab]))
 return {'source_zip_hash':ZIP_HASH,'sampled_original_archival_pages':len(selected),'retokenization_and_all_wordform_counts':'PASS'}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase3',required=True);p.add_argument('--phase4',required=True);p.add_argument('--source',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 zz=[verify_phase(a.phase3,a.phase4,x) for x in ('dev','test')]
 result={'schema':'PRHI_PHASE4_INDEPENDENT_VERIFICATION_V1','results':zz,'raw_source_check':raw_text_check(a.phase3,a.source)}
 out=Path(a.out);out.parent.mkdir(exist_ok=True,parents=True);out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
 print(json.dumps(result,indent=2))
