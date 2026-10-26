#!/usr/bin/env python3
"""Retrospective evaluation-mask sensitivity, with frozen existing trained models."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from audit_overlap import load_pages, index_ngrams, digest
from pilot_charlm import doc_rep, eval_novel_masks, normalize_weight_multipliers, build_model
from selection_reversal import compute_word_losses, by_issue, get_summary, bootstrap

ROOT=Path('/mnt/data/isiXhosa_publication_continuation')


def keep_word_mask(p,ix,n):
    words=p['tokens']; covered=bytearray(len(words))
    for i in range(max(0,len(words)-n+1)):
        matches=ix.get(digest(words[i:i+n]),())
        if any(iss!=p['issue'] for iss,dom in matches):
            covered[i:i+n]=b'\x01'*n
    return ~np.frombuffer(covered,dtype=np.uint8).astype(bool)


def main():
    pages=load_pages();train=[p for p in pages if p['split']=='train']
    indices={n:index_ngrams(train,n)[0] for n in [8,12,20]}
    docs=[doc_rep(p,indices[12]) for p in pages]
    tr_docs=[d for d in docs if d['split']=='train'];eval_docs=[d for d in docs if d['split']!='train']
    eval_novel_masks(eval_docs,indices[12])
    alpha={ch for d in tr_docs for ch in d['text']};alpha.add('?')
    factors=normalize_weight_multipliers(tr_docs)
    scored={}
    for policy in ('BALANCED','HARD'):
        model=build_model(tr_docs,policy,factors,alpha)
        scored[policy]=compute_word_losses(model,eval_docs,alpha)
        print('scored policy',policy,flush=True)
    original_masks={name:x['keep'].copy() for name,x in scored['BALANCED'].items()}
    output={'protocol':'MASK_SENSITIVITY_PROTOCOL.md','experimental_status':'RETROSPECTIVE, not independent','ns':{}}
    saved=json.loads((ROOT/'selection_reversal_results.json').read_text())
    for n in (8,12,20):
        masks={p['name']:keep_word_mask(p,indices[n],n) for p in pages if p['split']!='train'}
        for policy in scored:
            for name,vec in scored[policy].items():
                assert len(masks[name])==len(vec['keep'])
                if n==12:assert np.array_equal(masks[name],original_masks[name]),name
                vec['keep']=masks[name]
        output['ns'][str(n)]={}
        for split in ('validation','old_test'):
            sub=[d for d in eval_docs if d['split']==split]
            issue=by_issue(scored,sub); summary=get_summary(issue)
            b=bootstrap(issue,seed=20261022+n)
            diffs={dom:summary['BALANCED']['domains'][dom]['nonmatch']-summary['HARD']['domains'][dom]['nonmatch'] for dom in ('IMVO','ISIG')}
            delta=summary['BALANCED']['macro_nonmatch']-summary['HARD']['macro_nonmatch']
            fractions={dom:summary['BALANCED']['domains'][dom]['chars_match']/summary['BALANCED']['domains'][dom]['chars'] for dom in ('IMVO','ISIG')}
            if n==12:
                prev=saved['splits'][split]['summary']
                assert abs(prev['HARD']['macro_nonmatch']-summary['HARD']['macro_nonmatch'])<1e-7
                assert abs(prev['BALANCED']['macro_nonmatch']-summary['BALANCED']['macro_nonmatch'])<1e-7
            output['ns'][str(n)][split]={'delta_macro_nonmatch_BPC':delta,
                'delta_source_nonmatch_BPC':diffs,
                'fraction_of_chars_with_train_Ngram_overlap':fractions,
                'macro_all_delta':summary['BALANCED']['macro_all']-summary['HARD']['macro_all'],
                'bootstrap_95_CI':b['contrasts_BALANCED_minus_HARD'],
                'issues':b['issue_counts'],
                'criterion_stable_positive_each_domain':bool(all(diffs[d]>0 and b['contrasts_BALANCED_minus_HARD'][d+'_nonmatch']['ci_95'][0]>0 for d in diffs))}
            print('n',n,split,'fraction',fractions,'delta',delta,
                  'interval',b['contrasts_BALANCED_minus_HARD']['macro_nonmatch']['ci_95'],flush=True)
        (ROOT/'mask_sensitivity_results.json').write_text(json.dumps(output,indent=2))
    print('saved sensitivity results',flush=True)

if __name__=='__main__':main()
