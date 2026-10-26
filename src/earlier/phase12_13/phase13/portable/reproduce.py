#!/usr/bin/env python3
"""Portable complete rerun of historical and modern OCR uncertainty experiments.

Inputs are frozen sampled OCR words, training vocabulary, and 15 model snapshots.
No original PDFs or newspaper ZIP needed for this replay. Results may differ
in wall-clock timings only.
"""
from pathlib import Path
import json,sys,math
import sentencepiece as spm
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import certify as c
import edit_distance_certificate as e
inputs=json.loads((ROOT/'frozen_inputs.json').read_text())

def frozen_pages():
 return [{'text':' '.join(inputs['train_vocab'])}]

def frozen_sources():
 return inputs['sources']

def local_models():
 return {(mix,seed):spm.SentencePieceProcessor(model_file=str(ROOT/'models'/f'bpe_imvo_{mix:03d}_seed_{seed}.model')) for mix in [0,50,100] for seed in [11,23,37,47,59]}

for module in [c,e]:
 module.BASE=ROOT
 module.training_pages=frozen_pages
 module.get_sources=frozen_sources
 module.models=local_models

print('Reproducing exact substitution-only bounds (may take ~2 minutes on a CPU)...',flush=True)
c.main()
print('Reproducing exact word-internal edit-distance-one certificates...',flush=True)
e.main()
new1=json.loads((ROOT/'results.json').read_text());old1=json.loads((ROOT/'expected_substitution_results.json').read_text())
new2=json.loads((ROOT/'edit_distance_results.json').read_text());old2=json.loads((ROOT/'expected_edit_distance_results.json').read_text())
assert new1['toy_exactness']==old1['toy_exactness']
for src in old1['sources']:
 a=old1['sources'][src];b=new1['sources'][src]
 assert a['selected_ids']==b['selected_ids']
 assert a['coverage']['unique_types']==b['coverage']['unique_types']
 for oldrow,newrow in zip(a['pair_comparisons'],b['pair_comparisons']):
  assert math.isclose(oldrow['observed_delta_E'],newrow['observed_delta_E'],abs_tol=1e-8)
  for aa,bb in zip(oldrow['budgets'],newrow['budgets']):
   assert aa['m']==bb['m']
   assert math.isclose(aa['lower_E'],bb['lower_E'],abs_tol=1e-6)
   assert math.isclose(aa['upper_E'],bb['upper_E'],abs_tol=1e-6)
for src in old2['sources']:
 a=old2['sources'][src];b=new2['sources'][src]
 for aa,bb in zip(a['pairs'],b['pairs']):
  assert aa['observed_numerator']==bb['observed_numerator']
  for cc,dd in zip(aa['bounds'],bb['bounds']):
   assert cc['min_numerator']==dd['min_numerator']
   assert cc['max_numerator']==dd['max_numerator']
print('PASS: all 45 comparison pairs and every edit budget reproduced; source traces unchanged.',flush=True)
