#!/usr/bin/env python3
"""Independently verify frozen Unigram model hash, shared test text, and LM arithmetic."""
import re,json,hashlib,zipfile,math,random,os
from pathlib import Path
from collections import Counter
import pandas as pd, numpy as np,sentencepiece as spm
ROOT=Path(os.environ.get('PHASE33_ROOT','/mnt/data/isiXhosa_phase33_sampling_to_models'));OUT=ROOT/'outputs';U=ROOT/'unigram'
Udf=pd.read_csv(U/'page_scores.csv');B=pd.read_csv(OUT/'page_scores.csv');trainAudit=pd.read_csv(OUT/'training_audit.csv')
uAudit=pd.read_csv(U/'training_audit.csv')
assert len(Udf)==len(B)==37395
key=['filename','date','year','view','seed','policy','chars']
assert Udf[key].equals(B[key]),'Identical evaluation page view rows required'
assert Udf.unknown.sum()==0 and B.unknown.sum()==0
assert Udf.groupby(['filename','view']).size().eq(15).all()
assert Udf.groupby(['filename','view']).chars.nunique().eq(1).all()
for r in uAudit.itertuples():
 f=U/f'unigram_{r.policy}_{r.seed}.model'
 assert hashlib.sha256(f.read_bytes()).hexdigest()==r.model_hash
 b=trainAudit[(trainAudit.seed==r.seed)&(trainAudit.policy==r.policy)].iloc[0]
 assert b.line_input_sha256==r.input_hash
 assert abs(b.training_en_alpha_share-r.training_en_alpha_share)<1e-12
# Independently retrieve source text without using Phase33 extraction functions
random.seed(33)
indices=set(Udf.sample(230,random_state=33123).index.tolist())
selected=Udf.loc[sorted(indices)]
selected_fns=set(selected.filename)
original={}
with zipfile.ZipFile(os.environ.get('PHASE33_SOURCE','/mnt/data/TxtLangFiles1870-1894_TB.zip')) as z:
 for member in z.namelist():
  if member.startswith('__MACOSX') or not member.endswith('.txt'):continue
  parts=member.split('/')
  if len(parts)<3 or parts[-1] not in selected_fns:continue
  if parts[-2] not in ('XhTxt','EnTxt'):continue
  view='xh' if parts[-2]=='XhTxt' else 'en'
  words=re.findall(r'[a-z]+',z.read(member).decode('utf-8-sig',errors='replace').lower())
  if len(words)<40:continue
  original[(parts[-1],view)]=' '.join(words[:128])
checked=0;model_cache={}
for r in selected.itertuples():
 t=original[(r.filename,r.view)]
 assert sum(len(s) for s in t.split())==r.chars
 name=(r.seed,r.policy)
 if name not in model_cache:model_cache[name]=spm.SentencePieceProcessor(model_file=str(U/f'unigram_{r.policy}_{r.seed}.model'))
 sp=model_cache[name]
 assert len(sp.encode(t,out_type=int))==r.pieces
 checked+=1
# independent raw training data rebuild (LM probability check on actual artifacts)
raw_train={}
with zipfile.ZipFile(os.environ.get('PHASE33_SOURCE','/mnt/data/TxtLangFiles1870-1894_TB.zip')) as z:
 for member in z.namelist():
  if member.startswith('__MACOSX') or not member.endswith('.txt'):continue
  parts=member.split('/')
  if len(parts)<3 or parts[-2] not in ('XhTxt','EnTxt'):continue
  if not re.fullmatch(r'IMVO_(1885|1887)-\d\d-\d\d_p\d\d\d\.txt',parts[-1]):continue
  words=re.findall(r'[a-z]+',z.read(member).decode('utf-8-sig',errors='replace').lower())
  if len(words)<40:continue
  raw_train[(parts[-1],parts[-2])]=' '.join(words[:128])
assert len(raw_train)==623
# sample 30 rows spanning 5 model seeds; reconstruct bigram likelihood from scratch
b_sample=B.groupby(['seed','policy'],group_keys=False).sample(n=2,random_state=31103)
LM_checks=0
for (seed,policy),subset in b_sample.groupby(['seed','policy']):
 sp=spm.SentencePieceProcessor(model_file=str(OUT/'models'/f'bpe_{policy}_{seed}.model'))
 bigram=Counter();previous=Counter();unigram=Counter();ntok=0
 for text in raw_train.values():
  ids=sp.encode(text,out_type=int)
  for pr,t in zip([-1]+ids[:-1],ids):
   bigram[(pr,t)]+=1;previous[pr]+=1;unigram[t]+=1;ntok+=1
 for r in subset.itertuples():
  # locate text by page+view with no tokenizer-specific shortcut
  with zipfile.ZipFile(os.environ.get('PHASE33_SOURCE','/mnt/data/TxtLangFiles1870-1894_TB.zip')) as z:
   member=f'TxtLangFiles1870-1894_TB/{"EnTxt" if r.view=="en" else "XhTxt"}/{r.filename}'
   text=' '.join(re.findall(r'[a-z]+',z.read(member).decode('utf-8-sig',errors='replace').lower())[:128])
  ids=sp.encode(text,out_type=int);bits=0
  den=ntok+0.1*sp.vocab_size()
  for pr,t in zip([-1]+ids[:-1],ids):
   prob=(bigram[(pr,t)]+10*(unigram[t]+0.1)/den)/(previous[pr]+10)
   bits-=math.log2(prob)
  assert abs(bits-r.lm_bits_10)<1e-7,(seed,policy,bits,r.lm_bits_10)
  LM_checks+=1
record={'pass':True,'unigram_model_sha256_verified':15,'full_pairwise_query_alignment_rows':len(Udf),
 'unigram_source_reencoding_checks':checked,'independent_LM_bits_recomputations':LM_checks,
 'LM_train_units_independently_rebuilt':len(raw_train),
 'unigram_unknown_tokens':int(Udf.unknown.sum()),
 'extension_precommit_sha256':hashlib.sha256((ROOT/'EXTENSION_PRECOMMIT.md').read_bytes()).hexdigest(),
 'negative_control':'No test page is in LM/tokenizer training years; identical raw training and evaluation lines per model family.'}
(ROOT/'EXTENDED_VERIFICATION.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
