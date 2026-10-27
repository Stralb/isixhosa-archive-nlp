#!/usr/bin/env python3
"""Hold sampled training token multiset or corpus order fixed in controlled BPE training."""
import collections, hashlib, json, random, statistics, sys, tempfile, time
from pathlib import Path
import sentencepiece as spm
ROOT=Path(__file__).resolve().parent
EX=Path('/mnt/data/isiXhosa_historical_eval/isiXhosa_external_validation')
sys.path.insert(0,str(EX))
from common import training_pages,dev_pages,forms,alpha_len
from train_reference import sample_exact, BUDGET

assert BUDGET==421696
OUT=ROOT/'factorial_models'; OUT.mkdir(exist_ok=True)
old=EX/'models'
original_manifest=json.loads((old/'training_manifest.json').read_text())
assert original_manifest['mixes']==[0,0.5,1]
training=training_pages()
words={d:[w for p in training if p['source']==d for w in forms(p['text'])] for d in ['IMVO','ISIG']}
assert sum(map(alpha_len,words['ISIG']))==BUDGET
seen={w for s in words.values() for w in s}
dev=dev_pages()
val={d:[w for p in dev if p['source']==d for w in forms(p['text'])] for d in ['IMVO','ISIG']}
val_novel={d:[w for w in ww if w not in seen] for d,ww in val.items()}
alphabet=set(original_manifest['alphabet'])


def run(mix,mem,order):
    k=f'bpe_mix_{int(100*mix):03d}_membership_{mem}_order_{order}'
    n_imvo=int(round(BUDGET*mix));n_isig=BUDGET-n_imvo
    sample=sample_exact(words['IMVO'],n_imvo,mem*101+1)+sample_exact(words['ISIG'],n_isig,mem*101+2)
    assert sum(map(alpha_len,sample))==BUDGET
    membership_counts=collections.Counter(sample)
    assert sum(membership_counts.values())==len(sample)
    random.Random(order*101+3).shuffle(sample)
    prefix=str(OUT/k)
    with tempfile.TemporaryDirectory() as tmp:
        corpus=Path(tmp)/'corpus.txt'
        with corpus.open('w',encoding='utf8') as f:
            for i in range(0,len(sample),70):f.write(' '.join(sample[i:i+70])+'\n')
        spm.SentencePieceTrainer.train(input=str(corpus),model_prefix=prefix,
            model_type='bpe',vocab_size=2000,character_coverage=1.0,
            normalization_rule_name='identity',num_threads=2,hard_vocab_limit=False,
            bos_id=-1,eos_id=-1,pad_id=-1,unk_id=0,max_sentence_length=15000,
            shuffle_input_sentence=False,minloglevel=2)
    out={'mix':mix,'membership_seed':mem,'order_seed':order,'name':k,
         'membership_multiset_sha256':hashlib.sha256('\n'.join(f'{x}\t{membership_counts[x]}' for x in sorted(membership_counts)).encode()).hexdigest(),
         'membership_occurrences':len(sample),'membership_unique_types':len(membership_counts),
         'vocab_sha256':hashlib.sha256(Path(prefix+'.vocab').read_bytes()).hexdigest(),
         'model_sha256':hashlib.sha256(Path(prefix+'.model').read_bytes()).hexdigest(),
         'validation':{}}
    if mem==order and order in (11,23):
        oldname=f'bpe_imvo_{int(100*mix):03d}_seed_{mem}.vocab'
        out['byte_for_byte_original_vocab_reproduced']=Path(prefix+'.vocab').read_bytes()==(old/oldname).read_bytes()
    return out



def score_all_runs(rows):
    # Fixed support across all treated and saved baseline models, no model-specific UNK shortcuts.
    all_processors=[]
    for rr in rows:
        all_processors.append(spm.SentencePieceProcessor(model_file=str(OUT/(rr['name']+'.model'))))
    for rr in original_manifest['runs']:
        all_processors.append(spm.SentencePieceProcessor(model_file=str(old/(rr['name']+'.model'))))
    common={}
    for src in ('IMVO','ISIG'):
        eligible={}
        for w in set(val_novel[src]):
            ok=all(not c.isalpha() or c in alphabet for c in w)
            if ok:
                for p in all_processors:
                    if p.unk_id() in p.encode(w,out_type=int):
                        ok=False;break
            eligible[w]=ok
        ww=collections.Counter(w for w in val_novel[src] if eligible[w])
        denom=sum(n*alpha_len(w) for w,n in ww.items())
        assert denom>0
        common[src]={'counts':ww,'denom':denom,
                     'supported_occurrences':sum(ww.values()),
                     'total_unseen_occurrences':len(val_novel[src])}
    for rr in rows:
        p=spm.SentencePieceProcessor(model_file=str(OUT/(rr['name']+'.model')))
        for src in ('IMVO','ISIG'):
            obj=common[src]
            pieces=sum(n*len(p.encode(w,out_type=int)) for w,n in obj['counts'].items())
            rr['validation'][src]={'unseen_E':100*pieces/obj['denom'],
                                  'unseen_alpha':obj['denom'],
                                  'supported_unseen_occurrences':obj['supported_occurrences'],
                                  'all_unseen_occurrences':obj['total_unseen_occurrences']}
    return {s:{k:v for k,v in obj.items() if k!='counts'} for s,obj in common.items()}

def main():
    st=time.time();rows=[]
    for mix in [0,0.5,1]:
        for mem in ([11] if mix==0 else [11,23]):
            for order in [11,23,37]:
                result=run(mix,mem,order)
                rows.append(result)
                (ROOT/'sampling_factorial_checkpoint.json').write_text(json.dumps({'rows':rows},indent=2))
                print('FINISHED',result['name'],'types',result['membership_unique_types'],'vocab',result['vocab_sha256'][:12], 'replayed',result.get('byte_for_byte_original_vocab_reproduced'),'elapsed',round(time.time()-st,1),flush=True)
    common_support=score_all_runs(rows)
    doc={'common_support':common_support,'scope':'Two-factor membership x ordering intervention on retrospective 1887 newspaper validation',
         'precommit_sha256':hashlib.sha256((ROOT/'SAMPLING_FACTORIAL_PRECOMMIT.md').read_bytes()).hexdigest(),
         'runs':rows,'seconds':time.time()-st}
    (ROOT/'sampling_factorial_results.json').write_text(json.dumps(doc,indent=2))
    print('DONE',len(rows),round(time.time()-st,1),flush=True)

if __name__=='__main__':main()
