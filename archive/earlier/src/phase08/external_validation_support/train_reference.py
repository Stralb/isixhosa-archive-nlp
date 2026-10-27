"""Train fresh reference BPEs on Jan-Sep 1887 only; never read external corpora."""
import argparse, json, random, tempfile, hashlib
from pathlib import Path
import sentencepiece as spm
from common import ORIGINAL_ZIP,training_pages,dev_pages,forms,alpha_len

BUDGET=421696
SEEDS=[11,23,37,47,59] # new, independent replication; not the earlier study's exact models
MIXES=[0,0.5,1]

def sample_exact(flist,needed,seed):
    if needed==0:return []
    rng=random.Random(seed); seq=list(flist);rng.shuffle(seq)
    out=[]; rem=needed
    for w in seq:
        n=alpha_len(w)
        if n<=rem and n>0:out.append(w);rem-=n
        if rem==0: break
    if rem: raise ValueError(f'Could not sample {needed} characters; {rem} short')
    assert sum(map(alpha_len,out))==needed
    return out

def train(out,zip_path):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    pages=training_pages(zip_path)
    words={d:[w for p in pages if p['source']==d for w in forms(p['text'])] for d in ['IMVO','ISIG']}
    validation=dev_pages(zip_path)
    assert sum(map(alpha_len,words['ISIG']))==BUDGET, 'Unchanged 1887 reference budget check failed'
    shared_alpha=sorted(set(c for ws in words.values() for w in ws for c in w if c.isalpha()))
    reports=[]
    for seed in SEEDS:
        for mix in MIXES:
            n_imvo=int(round(BUDGET*mix)); n_isig=BUDGET-n_imvo
            sample=sample_exact(words['IMVO'],n_imvo,seed*101+1)+sample_exact(words['ISIG'],n_isig,seed*101+2)
            assert sum(map(alpha_len,sample))==BUDGET
            random.Random(seed*101+3).shuffle(sample)
            slug=f'bpe_imvo_{int(mix*100):03d}_seed_{seed}'
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'corpus.txt'
                with p.open('w',encoding='utf-8') as f:
                    for i in range(0,len(sample),70):f.write(' '.join(sample[i:i+70])+'\n')
                spm.SentencePieceTrainer.train(input=str(p),model_prefix=str(out/slug),model_type='bpe',vocab_size=2000,
                    character_coverage=1.0,normalization_rule_name='identity',num_threads=2,hard_vocab_limit=False,
                    bos_id=-1,eos_id=-1,pad_id=-1,unk_id=0,max_sentence_length=15000,
                    shuffle_input_sentence=False,minloglevel=2)
            model=spm.SentencePieceProcessor(model_file=str(out/(slug+'.model')))
            costs={}
            for d in ['IMVO','ISIG']:
                ww=[w for p in validation if p['source']==d for w in forms(p['text'])]
                total=sum(map(alpha_len,ww)); pieces=sum(len(model.encode(w,out_type=int)) for w in ww)
                costs[d]={'forms':len(ww),'chars':total,'pieces':pieces,'pieces_per_100_chars':100*pieces/total}
            reports.append({'seed':seed,'imvo_share':mix,'name':slug,'training_alpha_chars':BUDGET,'validation':costs,
                            'model_sha256':hashlib.sha256((out/(slug+'.model')).read_bytes()).hexdigest()})
            print(slug,'validation E IMVO/ISIG',round(costs['IMVO']['pieces_per_100_chars'],3),round(costs['ISIG']['pieces_per_100_chars'],3))
    (out/'training_manifest.json').write_text(json.dumps({'origin':'new independent reference, not original study models',
        'newspaper_zip':zip_path,'split':'Jan-Sep 1887 train, Oct-Dec 1887 validation; 1888 not accessed',
        'budget_alpha_chars':BUDGET,'model_type':'SentencePiece BPE','vocab_size':2000,
        'seeds':SEEDS,'mixes':MIXES,'alphabet':shared_alpha,'runs':reports},indent=2,ensure_ascii=False))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--zip',default=ORIGINAL_ZIP);a.add_argument('--out',default='models');args=a.parse_args();train(args.out,args.zip)
