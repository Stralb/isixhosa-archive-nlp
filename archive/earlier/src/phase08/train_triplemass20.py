#!/usr/bin/env python3
import collections,hashlib,json,random,sys,tempfile,time
from pathlib import Path
import sentencepiece as spm
import run_phase8 as r
ROOT=r.ROOT
orig=[w for p in r.training_pages_cached if p['source']=='IMVO' for w in r.forms(p['text'])]
TOTAL=3*r.BUDGET
assert TOTAL==1265088 and sum(map(r.alpha_len,orig))>TOTAL
CP=ROOT/'triplemass_training_checkpoint.json'
seeds=list(range(3001,3021))

def main():
  a,b=map(int,sys.argv[1:3]);rows=json.loads(CP.read_text())['runs'] if CP.exists() else []
  assert len(rows)==a
  for i in range(a,b):
    seed=seeds[i]
    sample=r.sample_exact(orig,TOTAL,seed*101+1)
    random.Random(seed*101+3).shuffle(sample)
    p=ROOT/f'triplemass_bpe_seed_{seed}'
    start=time.monotonic()
    with tempfile.TemporaryDirectory() as td:
      path=Path(td)/'input.txt'
      with path.open('w',encoding='utf8') as f:
        for j in range(0,len(sample),70):f.write(' '.join(sample[j:j+70])+'\n')
      spm.SentencePieceTrainer.train(input=str(path),model_prefix=str(p),model_type='bpe',vocab_size=2000,
         character_coverage=1.0,normalization_rule_name='identity',num_threads=2,
         hard_vocab_limit=False,bos_id=-1,eos_id=-1,pad_id=-1,unk_id=0,
         max_sentence_length=15000,shuffle_input_sentence=False,minloglevel=2)
    x={'seed':seed,'chars':TOTAL,'model_file':str(p.with_suffix('.model')),
       'training_walltime_s':time.monotonic()-start,'vocab_sha256':hashlib.sha256(p.with_suffix('.vocab').read_bytes()).hexdigest()}
    rows.append(x)
    CP.write_text(json.dumps({'protocol':'TRIPLE_MASS_REPLICATION_PROTOCOL.md','runs':rows},indent=2))
    print('TRAINED',len(rows),'/20',seed,round(x['training_walltime_s'],2),flush=True)
if __name__=='__main__':main()
