#!/usr/bin/env python3
"""Prospectively frozen Phase33B Unigram tokenizer-family check."""
import argparse,json,hashlib,time,shutil
from pathlib import Path
import numpy as np, pandas as pd
import sentencepiece as spm
from run_phase33 import parse_archive,training_units,sample_training,eval_units,hash_file,SEEDS,POLICIES,paired_boot

def main():
 p=argparse.ArgumentParser();p.add_argument('--zip',required=True);p.add_argument('--root',required=True);args=p.parse_args()
 root=Path(args.root);out=root/'outputs';u=root/'unigram';u.mkdir(exist_ok=True)
 audit=pd.read_csv(out/'training_audit.csv').set_index(['seed','policy'])
 pages=parse_archive(args.zip);train,units,n,p_en=training_units(pages);_,held=eval_units(pages)
 rows=[];meta=[];st=time.time()
 for seed in SEEDS:
  for policy in POLICIES:
   lines,rec=sample_training(train,units,n,p_en,policy,seed)
   assert rec['line_input_sha256']==audit.loc[(seed,policy),'line_input_sha256']
   prefix=u/f'unigram_{policy}_{seed}'
   tp=prefix.with_suffix('.tmp');tp.write_text('\n'.join(lines)+'\n',encoding='utf8')
   try:
    spm.SentencePieceTrainer.train(input=str(tp),model_prefix=str(prefix),vocab_size=2000,model_type='unigram',
      character_coverage=1.,hard_vocab_limit=False,normalization_rule_name='identity',
      bos_id=-1,eos_id=-1,pad_id=-1,unk_id=0,num_threads=1,max_sentence_length=4096,
      input_sentence_size=0,shuffle_input_sentence=False,split_by_whitespace=True,
      add_dummy_prefix=True,remove_extra_whitespaces=True)
   finally:tp.unlink(missing_ok=True)
   sp=spm.SentencePieceProcessor(model_file=str(prefix)+'.model')
   assert sp.vocab_size()>=1000
   meta.append(dict(seed=seed,policy=policy,model_hash=hash_file(str(prefix)+'.model'),vocab=sp.vocab_size(),
         input_hash=rec['line_input_sha256'],training_en_alpha_share=rec['training_en_alpha_share']))
   for a in held:
    toks=sp.encode(a['text'],out_type=int)
    rows.append(dict(filename=a['filename'],date=a['date'],year=a['year'],view=a['view'],seed=seed,
       policy=policy,chars=a['chars'],pieces=len(toks),unknown=sum(x==sp.unk_id() for x in toks),
       cohort='heldout_1888' if a['year']==1888 else 'later_1889_1894'))
   print('UNIGRAM',policy,seed,'vocab',sp.vocab_size(),'elapsed',round(time.time()-st,1),flush=True)
 dd=pd.DataFrame(rows);dd.to_csv(u/'page_scores.csv',index=False)
 mm=pd.DataFrame(meta);mm.to_csv(u/'training_audit.csv',index=False)
 scores=[]
 for cohort in ('heldout_1888','later_1889_1894'):
  for view in ('en','xh'):
   for a,b in (('P','W'),('P','M'),('M','W')):
    scores.append(paired_boot(dd,view,cohort,a,b,'pieces',seed=33800+int(view=='xh')+10*int(cohort=='later_1889_1894'),B=2000))
    if a=='P' and b=='W' and cohort=='later_1889_1894':
     scores.append(paired_boot(dd,view,cohort,a,b,'pieces',seed=33840, B=2000,by_year=True))
 cc=pd.DataFrame(scores);cc.to_csv(u/'paired_contrasts.csv',index=False)
 summary=[]
 for (co,view,policy,seed),g in dd.groupby(['cohort','view','policy','seed']):
  summary.append(dict(cohort=co,view=view,policy=policy,seed=seed,cost=100*g.pieces.sum()/g.chars.sum(),
          unknowns=int(g.unknown.sum())))
 ss=pd.DataFrame(summary);ss.to_csv(u/'per_seed_summary.csv',index=False)
 def get(co,view,contrast):
  return cc[(cc.cohort==co)&(cc.view==view)&(cc.contrast==contrast)&(cc.resample_unit=='date')].iloc[0]
 en=get('heldout_1888','en','P-W');rep=get('later_1889_1894','en','P-W');match=get('heldout_1888','en','P-M')
 xh=get('heldout_1888','xh','P-W');xhrep=get('later_1889_1894','xh','P-W')
 decisions={'H33B1':{'status':'SUPPORTED' if en.estimate>=.3 and en.interval_low>0 and rep.estimate>0 else 'NOT_SUPPORTED',
                      'heldout_1888':float(en.estimate),'later':float(rep.estimate)},
            'H33B2':{'status':'SUPPORTED' if -0.5<match.interval_low and match.interval_high<.5 else 'NOT_SUPPORTED',
                      'heldout_1888_P_minus_M':float(match.estimate),'ci':[float(match.interval_low),float(match.interval_high)]},
            'H33B3':{'status':'SUPPORTED' if xh.estimate<0 and xhrep.estimate<0 else 'NOT_SUPPORTED',
                      'heldout_1888':float(xh.estimate),'later':float(xhrep.estimate)}}
 (u/'RESULTS.json').write_text(json.dumps({'decisions':decisions,'model_count':len(meta),
             'page_score_rows':len(dd),'precommit_sha256':hash_file(root/'EXTENSION_PRECOMMIT.md')},indent=2)+'\n')
 print('RESULT',json.dumps(decisions,indent=2),flush=True)
if __name__=='__main__':main()
