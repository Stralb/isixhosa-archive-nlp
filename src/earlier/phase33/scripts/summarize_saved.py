#!/usr/bin/env python3
"""Complete Phase33 summaries without redoing successful tokenizer fits.
Reconstructs training audit deterministically from frozen sampler and confirms model hashes.
"""
from pathlib import Path
import os
import pandas as pd
from run_phase33 import parse_archive, training_units, sample_training, hash_file, SEEDS, POLICIES, summarize
ROOT=Path(os.environ.get('PHASE33_ROOT','/mnt/data/isiXhosa_phase33_sampling_to_models'))/'outputs'
source=os.environ.get('PHASE33_SOURCE','/mnt/data/TxtLangFiles1870-1894_TB.zip')
pages=parse_archive(source)
train,units,n,p_en=training_units(pages)
audits=[]
for seed in SEEDS:
 for policy in POLICIES:
  _,rec=sample_training(train,units,n,p_en,policy,seed)
  prefix=ROOT/'models'/f'bpe_{policy}_{seed}'
  assert prefix.with_suffix('.model').exists()
  rec.update(model_sha256=hash_file(prefix.with_suffix('.model')),
             vocab_size=2000,training_languages='IMVO_1885_and_1887',expected_P_en_word_share=p_en)
  audits.append(rec)
print('audits reconstructed',len(audits),flush=True)
score=pd.read_csv(ROOT/'page_scores.csv')
print('scores loaded',len(score),flush=True)
print(summarize(score.to_dict('records'),ROOT,audits))
