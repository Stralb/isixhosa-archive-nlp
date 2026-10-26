#!/usr/bin/env python3
"""Train and self-score 60 new 100% IMVO models for precommitted holdout of membership draws."""
import collections,hashlib,json,random,sys,time
import sentencepiece as spm
import run_phase8 as r
ROOT=r.ROOT
EXPECTED=(ROOT/'SELF_SCORE_PRECOMMIT_SHA256.txt').read_text().split()[0]
assert hashlib.sha256((ROOT/'SELF_SCORE_PRECOMMIT.md').read_bytes()).hexdigest()==EXPECTED
SEEDS=list(range(2001,2061))
CP=ROOT/'selfscore_training_checkpoint.json'
words=[w for p in r.training_pages_cached if p['source']=='IMVO' for w in r.forms(p['text'])]

def train(i):
    seed=SEEDS[i]
    row=r.train_one(r.issue_data(),'word_uniform',1.0,seed)
    # Exact own training sample: the original sampler; word order doesn't change multiset.
    sample=r.sample_exact(words,r.BUDGET,seed*101+1)
    freq=collections.Counter(sample)
    pp=spm.SentencePieceProcessor(model_file=str(r.MODELS/(row['name']+'.model')))
    items=list(freq.items())
    enc=pp.encode([w for w,n in items],out_type=int,num_threads=4)
    total=sum(n*len(z) for (w,n),z in zip(items,enc))
    assert sum(n*r.alpha_len(w) for w,n in items)==r.BUDGET
    row['self_train_E']=100*total/r.BUDGET
    row['unique_sampled_types_crosscheck']=len(freq)
    assert row['unique_sampled_types_crosscheck']==row['unique_sampled_types']
    return row

def main():
    start,stop=map(int,sys.argv[1:3]);rows=json.loads(CP.read_text())['runs'] if CP.exists() else []
    assert start==len(rows), (start,len(rows))
    t=time.monotonic()
    for i in range(start,stop):
        row=train(i);rows.append(row)
        CP.write_text(json.dumps({'precommit_sha256':EXPECTED,'runs':rows},indent=2))
        print('TRAINED',i+1,'/',len(SEEDS),'self_E',round(row['self_train_E'],4),'elapsed',round(time.monotonic()-t,2),flush=True)
    print('CHUNK FINISHED',len(rows),flush=True)
if __name__=='__main__':main()
