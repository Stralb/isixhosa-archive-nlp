#!/usr/bin/env python3
"""Refit representative models in a fresh directory using source-only samples.
Compares learned pieces, piece probabilities and encoding predictions, rather than
serialized .model hashes that may contain environment-dependent path metadata.
"""
import argparse,hashlib,json,sys,tempfile
from pathlib import Path
import sentencepiece as spm
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'Phase1'/'scripts'));sys.path.insert(0,str(ROOT/'Phase2'/'scripts'))
from audit_phase1 import parse
from factorial_sampler import build_pool,draw
from run_phase2 import train_sp,clean_validation

def fingerprint(sp):
 return hashlib.sha256(('\n'.join(f'{sp.id_to_piece(i)}\t{sp.get_score(i):.9g}' for i in range(sp.vocab_size()))+'\n').encode()).hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--archive',required=True);ap.add_argument('--root',required=True);a=ap.parse_args();root=Path(a.root)
 train,val,_,reads=parse(a.archive);pool=build_pool(train);windows,_,_=clean_validation(train,val)
 tests=[('bpe','PP',441),('bpe','WP',443),('unigram','PW',442),('unigram','WW',445)]
 rep=[]
 with tempfile.TemporaryDirectory(prefix='phase2_fresh_replay_') as folder:
  folder=Path(folder)
  for family,arm,seed in tests:
   label=f'{family}_{arm}_{seed}'
   lines,meta=draw(pool,arm,seed,500000)
   saved=json.loads((root/'Phase2'/'outputs'/'model_rows'/f'{label}.json').read_text())
   assert meta['sample_hash_sha256']==saved['sample_hash_sha256']
   prefix=folder/label
   ret=train_sp(lines,prefix,family)
   orig=spm.SentencePieceProcessor(model_file=str(root/'Phase2'/'outputs'/'models'/f'{label}.model'))
   assert ret.vocab_size()==orig.vocab_size()==2000
   assert fingerprint(ret)==fingerprint(orig),(label,'vocab mismatch')
   for w in windows:
    assert ret.encode(w['text'],out_type=int)==orig.encode(w['text'],out_type=int),(label,w['filename'])
   rep.append({'name':label,'sampling_sha256':meta['sample_hash_sha256'],'vocabulary_fingerprint_sha256':fingerprint(ret),
      'windows_same_encoding':len(windows),'vocab_size':ret.vocab_size(),'semantic_replay_PASS':True})
 result={'PASS':True,'refits':len(rep),'windows_reencoded':sum(r['windows_same_encoding'] for r in rep),'models':rep,'note':'Fresh independent process and output directory; compared learned pieces, scores, encodings, not path-dependent serialized bytes'}
 p=root/'Phase2'/'outputs'/'PHASE2_FRESH_REPLAY.json';p.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
