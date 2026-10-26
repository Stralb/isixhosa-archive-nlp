"""Freeze tokenizer selectors from 1887 validation, never from external corpora."""
import argparse,json,collections
from pathlib import Path
import sentencepiece as spm
from common import ORIGINAL_ZIP, training_pages,dev_pages,forms,alpha_len

def main(zip_path,model_dir):
    d=Path(model_dir); meta=json.loads((d/'training_manifest.json').read_text())
    train={w for p in training_pages(zip_path) for w in forms(p['text'])}
    val={dom:[w for p in dev_pages(zip_path) if p['source']==dom for w in forms(p['text'])] for dom in ['IMVO','ISIG']}
    scores={}
    for r in meta['runs']:
        sp=spm.SentencePieceProcessor(model_file=str(d/(r['name']+'.model')))
        per={}
        for dom,words in val.items():
            novel=[w for w in words if w not in train]
            nch=sum(map(alpha_len,novel));cost=sum(len(sp.encode(w)) for w in novel)
            per[dom]=100*cost/nch if nch else None
        scores[(r['seed'],r['imvo_share'])]=per
    seeds=meta['seeds']
    ranked=[]
    for mix in meta['mixes']:
        vals=[r for r in meta['runs'] if r['imvo_share']==mix]
        rawgap=sum(abs(r['validation']['IMVO']['pieces_per_100_chars']-
                       r['validation']['ISIG']['pieces_per_100_chars']) for r in vals)/len(vals)
        regrets=[]
        for seed in seeds:
            this=scores[(seed,mix)]
            regret_imvo=this['IMVO']-scores[(seed,1.0)]['IMVO']
            regret_isig=this['ISIG']-scores[(seed,0.0)]['ISIG']
            regrets.append(max(regret_imvo,regret_isig))
        ranked.append({'imvo_share':mix,'raw_abs_gap':rawgap,'worst_unseen_regret':sum(regrets)/len(regrets),
                       'unseen_regret_by_seed':regrets})
    best_raw=min(ranked,key=lambda a:(a['raw_abs_gap'],a['imvo_share']))
    best_op=min(ranked,key=lambda a:(a['worst_unseen_regret'],a['imvo_share']))
    frozen={'protocol':'PRE-EXTERNAL-ACCESS: selections based only on Oct-Dec 1887 validation',
            'training_reference':'Same-budget matched source-specific BPE models, fixed 2000 vocab and 5 seeds',
            'unseen_definition':'Surface form absent from all Jan-Sep 1887 newspaper training text',
            'raw_parity_selected':best_raw['imvo_share'],
            'opportunity_selected':best_op['imvo_share'],
            'domain_specialist_references':{'IMVO':1.0,'ISIG':0.0},
            'scores':ranked,
            'pass_metric_disagreement_gate':best_raw['imvo_share']!=best_op['imvo_share'],
            'external_test_policy':'Do not change tokenizer choice after downloading external corpora; report all mixtures too',
            'limits':'This is a signed specialist-relative excess-cost heuristic, not a social fairness metric.'}
    p=d/'selection_frozen.json';p.write_text(json.dumps(frozen,indent=2,ensure_ascii=False))
    print('Raw parity:',best_raw['imvo_share'],'Opportunity:',best_op['imvo_share'],
          'DIVERGE:',frozen['pass_metric_disagreement_gate'])
    return frozen
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--zip',default=ORIGINAL_ZIP);p.add_argument('--models',default='models');a=p.parse_args();main(a.zip,a.models)
