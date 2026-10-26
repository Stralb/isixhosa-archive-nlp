#!/usr/bin/env python3
"""Train-only determinism and allocation smoke tests; no models or held-out scores."""
import argparse, json, sys
from pathlib import Path
from audit_phase1 import parse
from factorial_sampler import ARMS,build_pool,draw,distributions

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    train,validation,testnames,stats=parse(a.archive)
    pool=build_pool(train)
    clean_pool=build_pool(train,exclude_reprints=True)
    support_pool=build_pool(train,common_support=True)
    assert {u['file']+':'+u['view'] for u in clean_pool}=={u['file']+':'+u['view'] for u in support_pool}
    results=[]
    for label,p in [('ordinary',pool),('reprint_excluded',clean_pool),('common_support_unfiltered',support_pool)]:
        for arm in ARMS:
            # 20k alphabetic chars suffices to test the full sampler behavior.
            lines,meta=draw(p,arm,seed=441,budget=20000)
            again,meta2=draw(p,arm,seed=441,budget=20000)
            assert meta==meta2 and lines==again
            assert len(lines)==meta['lines'] and len(lines)>40
            _,_,p_en=distributions(p,arm)
            assert 0<p_en<1
            meta.update(sample_type=label,p_en_probability=p_en,eligible_page_views=len(p),deterministic=True)
            results.append(meta)
    d={'all_pass':True,'source_content_access':'training only','source_archive_test_content_reads':0,
       'test_count':len(results),'training_pages':len(train),'validation_pages_not_used':len(validation),
       'arms':list(ARMS),'simulations':results}
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/'sampler_smoke.json').write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
    print('PASS',len(results),'paired deterministic train-only sampler checks')

if __name__=='__main__':main()
