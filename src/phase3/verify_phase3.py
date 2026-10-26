#!/usr/bin/env python3
"""Independent recomputation of ALL Phase3 encodings and paired issue bootstrap.
Separate parser and estimand implementation from run_phase3.py.
"""
from __future__ import annotations
import argparse,csv,json,re,hashlib,zipfile,collections,datetime
from pathlib import Path
import numpy as np
import pandas as pd
import sentencepiece as spm

FILENAME=re.compile(r'IMVO_(18(?:8[8-9]|9[0-4]))-(\d\d)-(\d\d)_p\d{3}\.txt$')
# Test-year filter explicitly 1888-1894 (1888,1889,1890,...,1894).
ARCHIVE_HASH='375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5'
PROTOCOL_HASH='bc443d883fa1c2b8eb09ced5f87d8f65b352d55512ef2c594b1b23e64d496bf4'
ALPHANUM=re.compile(r'[a-z]+')

def file_hash(file):return hashlib.sha256(Path(file).read_bytes()).hexdigest()

def readcsv(path):
    with open(path,encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--archive',required=True);p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.root);out=Path(a.output)
    assert file_hash(a.archive)==ARCHIVE_HASH
    assert file_hash(root/'Phase1'/'FROZEN_PROTOCOL.md')==PROTOCOL_HASH
    selection=json.loads((root/'Phase2'/'outputs'/'PHASE2_SELECTION_FREEZE.json').read_text())
    assert selection['selected_alpha']==100 and selection['heldout_content_reads']==0
    records=readcsv(root/'Phase2'/'outputs'/'model_training_manifest.csv')
    manifest=readcsv(out/'heldout_pageview_manifest.csv')
    assert len(records)==40 and len(manifest)==2493
    archive_pages={}
    with zipfile.ZipFile(a.archive) as z:
        for m in z.namelist():
            path=m.split('/')
            if len(path)<2 or path[-2] not in ('EnTxt','XhTxt'):continue
            match=FILENAME.fullmatch(path[-1])
            if not match:continue
            year=int(match[1])
            if year<1888 or year>1894:continue
            w=ALPHANUM.findall(z.read(m).decode('utf-8-sig',errors='replace').lower())
            if len(w)<40:continue
            v='en' if path[-2]=='EnTxt' else 'xh'
            r=w[:128]
            archive_pages[(path[-1],v)]={'text':' '.join(r),'year':year,'words':len(r),
                'fullwords':len(w),'chars':sum(map(len,r)),'issue':'IMVO_'+match[1]+'-'+match[2]+'-'+match[3]}
    assert len(archive_pages)==len(manifest)
    manifests={}
    for rec in manifest:
        key=(rec['filename'],rec['view'])
        assert key not in manifests and key in archive_pages
        src=archive_pages[key]
        assert rec['text_sha256']==hashlib.sha256(src['text'].encode()).hexdigest()
        assert int(rec['chars'])==src['chars'] and int(rec['words'])==src['words']
        assert int(rec['total_view_words'])==src['fullwords'] and rec['issue']==src['issue']
        assert (rec['period']=='1888')==(src['year']==1888)
        manifests[key]=rec
    groupby_label=collections.defaultdict(list)
    for rec in readcsv(out/'heldout_scores.csv'):
        groupby_label[f"{rec['family']}_{rec['arm']}_{rec['seed']}"].append(rec)
    assert len(groupby_label)==40 and all(len(v)==len(manifest) for v in groupby_label.values())
    max_piece_error=0;total_rows=0;total_unk=0
    for i,model_record in enumerate(records):
        label=model_record['model_name'];model=root/'Phase2'/'outputs'/'models'/f'{label}.model'
        assert file_hash(model)==model_record['model_sha256']
        sp=spm.SentencePieceProcessor(model_file=str(model))
        assert sp.vocab_size()==2000
        assert len(groupby_label[label])==2493
        seen=set()
        for r in groupby_label[label]:
            key=r['filename'],r['view'];assert key in archive_pages and key not in seen
            seen.add(key)
            src=archive_pages[key]
            assert r['issue']==src['issue'] and int(r['chars'])==src['chars']
            assert int(r['year'])==src['year']
            assert r['text_sha256']==hashlib.sha256(src['text'].encode()).hexdigest()
            tok=sp.encode(src['text'],out_type=int)
            assert len(tok)==int(r['pieces']), (label,key,len(tok),r['pieces'])
            unk=sum(t==sp.unk_id() for t in tok)
            assert unk==int(r['unknown'])
            total_unk+=unk;total_rows+=1
        assert len(seen)==2493
        if (i+1)%5==0:print('INDEPENDENT ENCODINGS',i+1,'/40',flush=True)
    assert total_rows==99720 and total_unk==0
    scores=pd.read_csv(out/'heldout_scores.csv')
    saved=pd.read_csv(out/'factorial_contrasts.csv')
    mx_abs=0;mx_ci=0
    rngs={'1888':np.random.default_rng(343401),'1889-94':np.random.default_rng(343402)}
    for period in ('1888','1889-94'):
        q=scores[scores.period==period]
        issues=sorted(q.issue.unique());n=len(issues)
        # Independent bootstrap: one multinomial of issue multiplicities, shared
        # across arms, seeds and both language views within the period.
        counts=rngs[period].multinomial(n,[1/n]*n,size=5000)
        for fam in ('bpe','unigram'):
            for v in ('en','xh'):
                ss=q[(q.family==fam)&(q.view==v)]
                denom=ss[(ss.seed==441)&(ss.arm=='PP')].groupby('issue').chars.sum().reindex(issues,fill_value=0).to_numpy()
                assert denom.sum()>0
                denom_draw=counts@denom
                assert (denom_draw>0).all()
                seed_est={};sample_est={}
                for arm in ('PP','PW','WP','WW'):
                    seed_est[arm]=np.zeros(5)
                    sample_est[arm]=np.zeros((5000,5))
                    for j,s in enumerate((441,442,443,444,445)):
                        sub=ss[(ss.arm==arm)&(ss.seed==s)]
                        by_iss=sub.groupby('issue').pieces.sum().reindex(issues,fill_value=0).to_numpy()
                        seed_est[arm][j]=100*by_iss.sum()/denom.sum()
                        sample_est[arm][:,j]=100*(counts@by_iss)/denom_draw
                def effects(z):
                    return dict(allocation=((z['WP']+z['WW'])-(z['PP']+z['PW']))/2,
                        within_view=((z['PW']+z['WW'])-(z['PP']+z['WP']))/2,
                        interaction=z['WW']-z['WP']-z['PW']+z['PP'],
                        WW_minus_PP=z['WW']-z['PP'])
                est=effects(seed_est);bootstrap=effects(sample_est)
                for c in est:
                    ref=saved[(saved.family==fam)&(saved.period==period)&(saved.view==v)&(saved.contrast==c)].iloc[0]
                    point=est[c].mean()
                    ci=np.percentile(bootstrap[c].mean(axis=1),[2.5,97.5])
                    mx_abs=max(mx_abs,abs(point-ref['mean']))
                    mx_ci=max(mx_ci,abs(ci[0]-ref['ci_low']),abs(ci[1]-ref['ci_high']))
                    assert abs(point-ref['mean'])<1e-11
                    assert max(abs(ci[0]-ref['ci_low']),abs(ci[1]-ref['ci_high']))<1e-10
    # independently compute source-view counts for 1888 and late years.
    for per in ('1888','1889-94'):
        rows=[rec for rec in manifest if rec['period']==per]
        assert len({r['issue'] for r in rows})==(52 if per=='1888' else 306)
    assert len(saved)==32
    result=dict(PASS=True,verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                source_archive_sha256=file_hash(a.archive),phase1_protocol_sha256=PROTOCOL_HASH,
                model_fingerprints=40,pageview_source_hashes=len(manifest),saved_predictions_recomputed=total_rows,
                max_abs_piece_error=max_piece_error,unknown_tokens=total_unk,
                issue_bootstrap_recomputed_5000=True,bootstrap_draws_per_period=5000,
                factorial_contrasts_verified=len(saved),max_point_difference=mx_abs,max_ci_difference=mx_ci,
                existing_validation_only_selection_lock_verified=True,
                confirmatory_scope='conditional followup: prior studies examined 1888 onward')
    (out/'PHASE3_INDEPENDENT_VERIFICATION.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('INDEPENDENT_VERIFICATION',json.dumps(result,indent=2),flush=True)

if __name__=='__main__': main()
