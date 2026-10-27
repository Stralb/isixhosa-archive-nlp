#!/usr/bin/env python3
"""Explicitly post-hoc: decomposes archive page positions and model-family interactions."""
from pathlib import Path
import numpy as np,pandas as pd,json,os
from run_phase33 import parse_archive,training_units,paired_boot
root=Path(os.environ.get('PHASE33_ROOT','/mnt/data/isiXhosa_phase33_sampling_to_models'));o=root/'outputs';u=root/'unigram'
archive=os.environ.get('PHASE33_SOURCE','/mnt/data/TxtLangFiles1870-1894_TB.zip')
pages=parse_archive(archive);train,units,n,p_en=training_units(pages)
base=pd.DataFrame([dict(page=r['page'], date=r['date'], eligible_xh=int(n[i,0]), eligible_en=int(n[i,1]), eligible_total=int(n[i].sum())) for i,r in enumerate(train)])
base['P_en_mass']=base.eligible_en/base.eligible_total/len(base)
base['W_en_mass']=base.eligible_en/base.eligible_total.sum()
base['P_page_prob']=1/len(base)
base['W_page_prob']=base.eligible_total/base.eligible_total.sum()
pos=base.groupby('page').agg(pages=('date','size'),eligible_words=('eligible_total','sum'),
 page_prob_P=('P_page_prob','sum'),page_prob_W=('W_page_prob','sum'),
 En_prob_P=('P_en_mass','sum'),En_prob_W=('W_en_mass','sum'))
pos['en_share_within_page_P']=pos.En_prob_P/pos.page_prob_P
pos['en_share_within_page_W']=pos.En_prob_W/pos.page_prob_W
pos['contribution_W_minus_P']=pos.En_prob_W-pos.En_prob_P
pos.reset_index().to_csv(root/'POSTHOC_PAGE_MECHANISM.csv',index=False)
print('position details:\n',pos.round(4))
print('word english P/W',base.P_en_mass.sum(),base.W_en_mass.sum(), 'contrast',base.W_en_mass.sum()-base.P_en_mass.sum())
b=pd.read_csv(o/'page_scores.csv');v=pd.read_csv(u/'page_scores.csv')
mergecols=['filename','date','year','view','seed','policy','chars']
b['cohort']=np.where(b.year==1888,'heldout_1888','later_1889_1894')
d=b.merge(v,on=mergecols+['cohort'],suffixes=('_bpe','_unigram'),validate='one_to_one')
assert len(d)==37395
obs=d[mergecols+['cohort']].copy();obs['pieces']=d.pieces_unigram-d.pieces_bpe
rows=[]
for cohort in ('heldout_1888','later_1889_1894'):
 for view in ('en','xh'):
  for a,bc in [('P','W'),('P','M'),('M','W')]:
   rows.append(paired_boot(obs,view,cohort,a,bc,'pieces',seed=33900+int(view=='xh')+10*int(cohort=='later_1889_1894'),B=2000))
inter=pd.DataFrame(rows);inter.to_csv(root/'POSTHOC_TOKENIZER_FAMILY_INTERACTION.csv',index=False)
print('interaction U minus BPE by contrast:')
print(inter[['cohort','view','contrast','estimate','interval_low','interval_high']].round(4).to_string(index=False))
per=[]
for name,df in [('BPE',b),('UNIGRAM',v)]:
 for (y,view,policy,seed),grp in df.groupby(['year','view','policy','seed']):
  r=dict(family=name,year=y,view=view,policy=policy,seed=seed,
         issues=grp.date.nunique(),pages=len(grp),cost=100*grp.pieces.sum()/grp.chars.sum())
  if name=='BPE':r['bpc_10']=grp.lm_bits_10.sum()/grp.chars.sum()
  per.append(r)
per=pd.DataFrame(per);per.to_csv(root/'POSTHOC_YEARLY_MODEL_SCORES.csv',index=False)
# fixed evaluation membership determines identical positive denominators across policies
piv=per.pivot(index=['family','year','view','seed'],columns='policy',values='cost').reset_index()
piv['P_minus_W']=piv['P']-piv['W'];piv['P_minus_M']=piv['P']-piv['M']
piv.groupby(['family','year','view'],as_index=False).agg(avg_P_minus_W=('P_minus_W','mean'),avg_P_minus_M=('P_minus_M','mean')).to_csv(root/'POSTHOC_YEARLY_CONTRASTS.csv',index=False)
print('yearly PW gaps\n',piv.groupby(['family','year','view']).P_minus_W.mean().round(3).to_string())
tr=pd.read_csv(o/'training_audit.csv');pivt=tr.pivot(index='seed',columns='policy',values='training_en_alpha_share');pivt['P_minus_M']=pivt.P-pivt.M
print('P-M actual share differences per seed:',pivt['P_minus_M'].round(4).tolist())
meta={'dataset_train_issues':len(set(r['date'] for r in train)),
 'page_uniform_english_word_probability':float(base.P_en_mass.sum()),
 'word_weighted_english_word_probability':float(base.W_en_mass.sum()),
 'word_share_difference':float(base.W_en_mass.sum()-base.P_en_mass.sum()),
 'source_years':'IMVO 1885 and 1887',
 'page3_contribution_to_word_minus_page':float(pos.loc[3,'contribution_W_minus_P']),
 'posthoc_label':True}
(root/'POSTHOC_DIAGNOSTICS.json').write_text(json.dumps(meta,indent=2)+'\n')
