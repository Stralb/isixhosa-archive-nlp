#!/usr/bin/env python3
"""Exploratory link to previous source-composition BPE work; not Study 1 replication."""
import argparse,collections,csv,hashlib,json
from pathlib import Path
import numpy as np
import sentencepiece as spm
SEED=26024
BUDGET=60000

def fields(data,path):
    with open(path,'w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=data[0].keys());w.writeheader();w.writerows(data)

def prepare(train,source,seed,out):
    rng=np.random.default_rng(SEED+10*seed+(500 if source=='IMVO' else 100))
    sel=[w for w in train if w['source']==source]
    rng.shuffle(sel)
    written=0; rows=[]
    for p in sel:
        line=[]
        for token in p['tokens']:
            letters=sum(ch.isalpha() for ch in token)
            remain=BUDGET-written
            if remain<=0:break
            if letters<=remain:
                line.append(token);written+=letters
            else:
                # One final truncated token (feasibility); recorded. Use only needed alphabetic chars.
                cutoff=''.join([ch for ch in token if ch.isalpha()][:remain])
                line.append(cutoff);written+=remain
                break
        if line:rows.append(' '.join(line))
        if written==BUDGET:break
    assert written==BUDGET,(source,seed,written)
    path=out/f'train_{source}_seed{seed}.txt'
    path.write_text('\n'.join(rows)+'\n',encoding='utf8')
    return path,hashlib.sha256(path.read_bytes()).hexdigest(),len(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--windows',default=str(Path(__file__).parent/'windows.jsonl'));ap.add_argument('--output',default=str(Path(__file__).parent));args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    wins=[json.loads(l) for l in open(args.windows,encoding='utf8')]
    train=[w for w in wins if w['year'] in (1885,1887)]; test=[w for w in wins if w['year']==1888]
    rows=[];trained=[]
    for seed in range(3):
        models={}
        for source in ['IMVO','ISIG']:
            input,sha,n=prepare(train,source,seed,out)
            prefix=str(out/f'BPE_{source}_seed{seed}')
            spm.SentencePieceTrainer.train(input=str(input),model_prefix=prefix,model_type='bpe',vocab_size=1000,character_coverage=1.0,normalization_rule_name='identity',hard_vocab_limit=False,unk_id=0,bos_id=-1,eos_id=-1,pad_id=-1,shuffle_input_sentence=False)
            model=spm.SentencePieceProcessor(model_file=prefix+'.model')
            models[source]=model
            trained.append(dict(source=source,seed=seed,training_characters=BUDGET,training_lines=n,sha256=sha,actual_vocab=model.vocab_size()))
        for w in test:
            text=w['text'];c=sum(ch.isalpha() for ch in text)
            d={key:w[key] for key in ['file','source','year','issue']};d.update(seed=seed,alphabetic_chars=c)
            for source in ('IMVO','ISIG'):
                pieces=len(models[source].encode(text,out_type=int))
                d[f'{source}_pieces']=pieces
                d[f'{source}_E']=100*pieces/c
            rows.append(d)
    fields(rows,out/'tokenizer_page_scores.csv')
    fields(trained,out/'tokenizer_training_audit.csv')
    summary=[]
    for seed in range(3):
        for source in ('IMVO','ISIG'):
            s=[r for r in rows if r['seed']==seed and r['source']==source]
            def cost(model):return 100*sum(r[f'{model}_pieces'] for r in s)/sum(r['alphabetic_chars'] for r in s)
            a,b=cost('IMVO'),cost('ISIG')
            summary.append(dict(seed=seed,test_source=source,Imvo_trained_pieces_per100chars=a,Isig_trained_pieces_per100chars=b,benefit_of_correct_source=(b-a) if source=='IMVO' else (a-b),n_pages=len(s),n_issues=len(set(r['issue'] for r in s))))
    fields(summary,out/'tokenizer_summary.csv')
    # Issue and training-seed joint bootstrap. Each bootstrap draws one of the 3 trained seed-pairs, then resamples issues per source.
    rng=np.random.default_rng(SEED)
    sample_ci={}
    for source in ('IMVO','ISIG'):
        ds=collections.defaultdict(list)
        for r in rows:
            if r['source']==source:ds[(r['seed'],r['issue'])].append(r)
        per={}
        for seed in range(3):
            per[seed]={}
            for (se,iss),items in ds.items():
                if se!=seed:continue
                x=sum(r['alphabetic_chars'] for r in items)
                a=100*sum(r['IMVO_pieces'] for r in items)/x
                b=100*sum(r['ISIG_pieces'] for r in items)/x
                per[seed][iss]=(b-a) if source=='IMVO' else (a-b)
        vals=[]
        for _ in range(2000):
            seed=int(rng.integers(0,3));iscores=list(per[seed].values());vals.append(float(np.mean(rng.choice(iscores,len(iscores),replace=True))))
        sample_ci[source]=dict(mean=float(np.mean([r['benefit_of_correct_source'] for r in summary if r['test_source']==source])),bootstrap_95ci=[float(x) for x in np.quantile(vals,[.025,.975])],seeds_positive=int(sum(r['benefit_of_correct_source']>0 for r in summary if r['test_source']==source)))
    D=[]
    for seed in range(3):
        vs={}
        for model in ['IMVO','ISIG']:
            ss={}
            for source in ['IMVO','ISIG']:
                selected=[r for r in rows if r['seed']==seed and r['source']==source]
                ss[source]=100*sum(r[f'{model}_pieces'] for r in selected)/sum(r['alphabetic_chars'] for r in selected)
            vs[model]=ss['IMVO']-ss['ISIG']
        D.append(dict(seed=seed,D_Imvo_trained=vs['IMVO'],D_Isig_trained=vs['ISIG'],D_source_shift=vs['IMVO']-vs['ISIG']))
    fields(D,out/'tokenizer_disparity_shift.csv')
    result=dict(n=3,train_audit=trained,costs=summary,within_source_benefits=sample_ci,source_disparity_shift=D)
    (out/'tokenizer_link_results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(benefits=sample_ci,disparity=D),indent=2),flush=True)

if __name__=='__main__':main()
