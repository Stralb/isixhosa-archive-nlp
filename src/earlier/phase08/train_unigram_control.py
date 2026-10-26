#!/usr/bin/env python3
import hashlib,json,random,sys,tempfile,time
from pathlib import Path
import sentencepiece as spm
import run_phase8 as r
ROOT=r.ROOT
EXPECTED=(ROOT/'UNIGRAM_CONTROL_PRECOMMIT_SHA256.txt').read_text().split()[0]
assert hashlib.sha256((ROOT/'UNIGRAM_CONTROL_PRECOMMIT.md').read_bytes()).hexdigest()==EXPECTED
PARAMS=[(mix,seed) for mix in (.5,1.) for seed in r.SEEDS]
CP=ROOT/'unigram_training_checkpoint.json'
PAGES=r.training_pages_cached
WORDS={src:[w for pg in PAGES if pg['source']==src for w in r.forms(pg['text'])] for src in ('IMVO','ISIG')}


def fit(mix,seed,duplicate=False):
    imvo=round(r.BUDGET*mix)
    sample=r.sample_exact(WORDS['IMVO'],imvo,seed*101+1)+r.sample_exact(WORDS['ISIG'],r.BUDGET-imvo,seed*101+2)
    assert sum(map(r.alpha_len,sample))==r.BUDGET
    random.Random(seed*101+3).shuffle(sample)
    name=f'unigram_mix{int(mix*100):03d}_draw{seed}'+('_refit' if duplicate else '')
    p=ROOT/(name)
    t=time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
      inp=Path(tmp)/'train.txt'
      with inp.open('w',encoding='utf8') as f:
        for i in range(0,len(sample),70):f.write(' '.join(sample[i:i+70])+'\n')
      spm.SentencePieceTrainer.train(input=str(inp),model_prefix=str(p),
        model_type='unigram',vocab_size=2000,character_coverage=1.0,
        normalization_rule_name='identity',num_threads=2,hard_vocab_limit=False,
        bos_id=-1,eos_id=-1,pad_id=-1,unk_id=0,max_sentence_length=15000,
        shuffle_input_sentence=False,minloglevel=2)
    row={'mix':mix,'seed':seed,'name':name,'model_file':str(p.with_suffix('.model')),
      'vocab_sha256':hashlib.sha256(p.with_suffix('.vocab').read_bytes()).hexdigest(),
      'training_walltime_s':time.monotonic()-t}
    return row

def main():
    start,stop=map(int,sys.argv[1:3]);rows=json.loads(CP.read_text())['runs'] if CP.exists() else []
    assert len(rows)==start
    for i in range(start,stop):
      mix,seed=PARAMS[i]
      row=fit(mix,seed);rows.append(row)
      CP.write_text(json.dumps({'precommit_sha256':EXPECTED,'runs':rows},indent=2))
      print('UNIGRAM',len(rows),'/40',mix,seed,'sec',round(row['training_walltime_s'],2),flush=True)
    if stop==40:
      row=fit(1.,1001,duplicate=True)
      orig=next(v for v in rows if v['mix']==1. and v['seed']==1001)
      a=row['vocab_sha256']==orig['vocab_sha256']
      (ROOT/'unigram_determinism.json').write_text(json.dumps({'original':orig,'repeat':row,'byte_identical_vocab':a},indent=2))
      print('REPLAY UNIGRAM SAME VOCAB',a,flush=True)
if __name__=='__main__':main()
