#!/usr/bin/env python3
"""Phase-4 F6 TRAIN ONLY. No post-1887 text is loaded or inspected."""
from __future__ import annotations
import argparse, csv, hashlib, json, sys, collections, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Phase1'/'scripts'))
sys.path.insert(0,str(ROOT/'Phase2'/'scripts'))
from audit_phase1 import parse,enumerate_units,K,WINDOW
from factorial_sampler import build_pool,distributions,draw
from run_phase2 import train_sp
import sentencepiece as spm

SOURCE_HASH='375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5'
PROTOCOL_HASH='bc443d883fa1c2b8eb09ced5f87d8f65b352d55512ef2c594b1b23e64d496bf4'
PRECOMMIT_HASH='1fb49e638a51cedc7f8157743d2c2d90ba65003f4d049990cf8d23c30f00714e'
SEEDS=(441,442,443,444,445)
ARMS=('PP','WW')
CONDITIONS=('filtered','support_control')
BUDGET=500000

def sha_file(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def write_csv(path,rows,fields):
 with open(path,'w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def repeat_owners(train):
 units,_=enumerate_units(train)
 owners=collections.defaultdict(set)
 for u in units:
  words=u['words']
  for i in range(len(words)-K+1):owners[tuple(words[i:i+K])].add(u['issue'])
 return {gram for gram,issues in owners.items() if len(issues)>1}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--archive',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
 out=Path(args.output); (out/'models').mkdir(parents=True,exist_ok=True)
 (out/'model_rows').mkdir(parents=True,exist_ok=True)
 assert sha_file(args.archive)==SOURCE_HASH
 assert sha_file(ROOT/'Phase1'/'FROZEN_PROTOCOL.md')==PROTOCOL_HASH
 assert sha_file(ROOT/'Phase4'/'PRECOMMIT.md')==PRECOMMIT_HASH
 train,val,_,access=parse(args.archive)
 assert access['heldout_view_content_reads']==0
 filtered=build_pool(train,exclude_reprints=True)
 control=build_pool(train,common_support=True)
 key=lambda u:(u['file'],u['view'])
 assert {key(u) for u in filtered}=={key(u) for u in control}
 assert len(filtered)==len(control)>300
 fdict={key(u):u for u in filtered};cdict={key(u):u for u in control}
 assert all(fdict[k]['words']==cdict[k]['words'] and len(fdict[k]['starts'])<=len(cdict[k]['starts']) for k in fdict)
 forbidden=repeat_owners(train)
 assert len(forbidden)>10000
 support_rows=[]
 for k in sorted(fdict):
  f,c=fdict[k],cdict[k]
  support_rows.append(dict(filename=f['file'],issue=f['issue'],view=f['view'],nwords=f['nwords'],
    clean_windows=len(f['starts']),all_windows=len(c['starts']),clean_fraction=len(f['starts'])/len(c['starts'])))
 write_csv(out/'support_manifest.csv',support_rows,list(support_rows[0]))
 units,_=enumerate_units(train)
 source_keys={key(u) for u in units}
 dropped=source_keys-{key(u) for u in filtered}
 byview=lambda keys,v:sum(x[1]==v for x in keys)
 summary={'phase':'4','archive_sha256':SOURCE_HASH,'phase1_protocol_sha256':PROTOCOL_HASH,'phase4_precommit_sha256':PRECOMMIT_HASH,
  'read_heldout_content':False,'training_pages':len(train),'training_issues':len({r['issue'] for r in train}),
  'original_eligible_pageviews':len(units),'common_support_pageviews':len(filtered),'dropped_pageviews':len(dropped),
  'dropped_english_pageviews':byview(dropped,'en'),'dropped_isixhosa_pageviews':byview(dropped,'xh'),
  'retained_english_pageviews':byview(fdict,'en'),'retained_isixhosa_pageviews':byview(fdict,'xh'),
  'cross_issue_repeated_13gram_types':len(forbidden),
  'clean_window_count':sum(len(u['starts']) for u in filtered),
  'control_window_count':sum(len(u['starts']) for u in control),
  'matched_support_SHA256':hashlib.sha256(('\n'.join(f'{x[0]}::{x[1]}' for x in sorted(fdict))+'\n').encode()).hexdigest(),
  'arm_expected_en_share':{arm:distributions(filtered,arm)[2] for arm in ARMS}}
 (out/'support_audit.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
 print('COMMON SUPPORT',summary,flush=True)
 models=[]
 t0=time.time()
 for cond in CONDITIONS:
  pool=filtered if cond=='filtered' else control
  for arm in ARMS:
   for seed in SEEDS:
    name=f'bpe_{cond}_{arm}_{seed}'
    prefix=out/'models'/name
    metadata=out/'model_rows'/f'{name}.json'
    lines,audit=draw(pool,arm,seed,budget=BUDGET)
    assert len(lines)>500
    ban_count=0
    for line in lines:
     w=line.split()
     ban_count+=sum(tuple(w[i:i+K]) in forbidden for i in range(len(w)-K+1))
    if cond=='filtered':assert ban_count==0,(name,ban_count)
    else:assert ban_count>=0
    if metadata.exists():
     rec=json.loads(metadata.read_text());assert rec['sample_hash_sha256']==audit['sample_hash_sha256']
     assert rec['model_sha256']==sha_file(str(prefix)+'.model')
     assert rec['vocab_sha256']==sha_file(str(prefix)+'.vocab')
    else:
     sp=train_sp(lines,prefix,'bpe')
     rec={**audit,'family':'bpe','condition':cond,'model_name':name,'model_sha256':sha_file(str(prefix)+'.model'),
          'vocab_sha256':sha_file(str(prefix)+'.vocab'),'vocab_size':sp.vocab_size(),
          'training_only_repeated_shingles_found':ban_count,'source_support_hash':summary['matched_support_SHA256'],
          'archive_sha256':SOURCE_HASH,'precommit_sha256':PRECOMMIT_HASH,
          'training_text_included_in_package':False}
     metadata.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
    assert rec['vocab_size']==2000
    models.append(rec)
    print('TRAIN/REUSE',name,'n',len(models),'of',20,'en_share',round(audit['en_alpha_share'],4),
      'repeat_13grams',ban_count,'elapsed_s',round(time.time()-t0,1),flush=True)
 fields=['model_name','family','condition','arm','seed','budget_target','actual_alpha_chars','en_alpha_chars',
 'en_alpha_share','lines','views_drawn_en','views_drawn_xh','unique_lines','source_issues_represented',
 'training_only_repeated_shingles_found','sample_hash_sha256','model_sha256','vocab_sha256','vocab_size',
 'source_support_hash','archive_sha256','precommit_sha256','training_text_included_in_package']
 write_csv(out/'model_training_manifest.csv',models,fields)
 assert len(models)==20
 result={'training_complete':True,'model_count':len(models),'filtered_models':sum(m['condition']=='filtered' for m in models),
   'common_support_controls':sum(m['condition']=='support_control' for m in models),
   'all_same_page_view_support':True,'all_filtered_have_no_repeated_13grams':all(
      x['training_only_repeated_shingles_found']==0 for x in models if x['condition']=='filtered'),
   'equal_budget_range':[min(m['actual_alpha_chars'] for m in models),max(m['actual_alpha_chars'] for m in models)],
   'model_manifest_sha256':sha_file(out/'model_training_manifest.csv'),'support_manifest_sha256':sha_file(out/'support_manifest.csv'),
   'test_content_read':False,'archive_sha256':SOURCE_HASH,'precommit_sha256':PRECOMMIT_HASH}
 (out/'TRAINING_LOCK.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print('TRAINING_LOCK',result,flush=True)

if __name__=='__main__':main()
