#!/usr/bin/env python3
import collections,hashlib,json,random,tempfile,time
from pathlib import Path
import sentencepiece as spm
import run_phase8 as r
p=r.ROOT
wordlist=[w for a in r.training_pages_cached if a['source']=='IMVO' for w in r.forms(a['text'])]
N=sum(r.alpha_len(w) for w in wordlist)
assert N==1406493
runs=[]
for tag,needed in [('triple_mass',3*r.BUDGET),('full_train_pool',N)]:
    if needed==N: sample=list(wordlist)
    else: sample=r.sample_exact(wordlist,needed,2001*101+1)
    random.Random(2001*101+3).shuffle(sample)
    prefix=p/(tag+'_imvo_bpe')
    start=time.monotonic()
    with tempfile.TemporaryDirectory() as td:
        f=Path(td)/'train.txt'
        with f.open('w',encoding='utf8') as out:
            for i in range(0,len(sample),70):out.write(' '.join(sample[i:i+70])+'\n')
        spm.SentencePieceTrainer.train(input=str(f),model_prefix=str(prefix),
            model_type='bpe',vocab_size=2000,character_coverage=1.0,
            normalization_rule_name='identity',num_threads=2,hard_vocab_limit=False,
            bos_id=-1,eos_id=-1,pad_id=-1,unk_id=0,max_sentence_length=15000,
            shuffle_input_sentence=False,minloglevel=2)
    d={'tag':tag,'characters':needed,'walltime_s':time.monotonic()-start,
       'model_file':str(prefix.with_suffix('.model')),
       'vocab_sha256':hashlib.sha256(prefix.with_suffix('.vocab').read_bytes()).hexdigest()}
    runs.append(d)
    print('TRAINED',tag,'walltime',d['walltime_s'],'sha',d['vocab_sha256'][:12],flush=True)
(p/'large_single_training.json').write_text(json.dumps({'protocol':'LARGE_SINGLE_BASELINE_PROTOCOL.md (post hoc control)','runs':runs},indent=2))
