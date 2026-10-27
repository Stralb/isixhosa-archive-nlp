#!/usr/bin/env python3
"""Independent aggregation/provenance checks for 1889-1894 future test."""
from pathlib import Path
import collections,csv,hashlib,json,random,zipfile
import numpy as np
import pandas as pd
import sentencepiece as spm
D=Path(__file__).parent
archive=Path('/mnt/data/TxtLangFiles1870-1894_TB.zip')
audit=json.loads((D/'future_sampling_audit.json').read_text())
window_file=D/'future_1889_1894_windows.jsonl'
if window_file.exists():
    W=[json.loads(l) for l in open(window_file,encoding='utf8')]
    assert hashlib.sha256(window_file.read_bytes()).hexdigest()==audit['window_file_sha256']
    assert len(W)==740 and len({w['file'] for w in W})==740
    assert len({w['text'] for w in W})==740
    assert {w['year'] for w in W}==set(range(1889,1895))
    assert all(len(w['tokens'])==90 and w['text']==' '.join(w['tokens']) for w in W)
    assert all(w['file'].startswith(f'IMVO_{w["year"]}') for w in W)
else:
    W=[]  # archive text deliberately excluded; run phase27_future_year.py to regenerate
with zipfile.ZipFile(archive) as z:
    names={p.rsplit('/',1)[-1] for p in z.namelist() if '/XhTxt/' in p and '__MACOSX' not in p}
assert all(w['file'] in names for w in W)
assert [audit['windows_per_year'][str(y)] for y in range(1889,1895)]==[124,128,121,120,118,129]
S=pd.read_csv(D/'future_page_scores.csv')
I=pd.read_csv(D/'future_issue_scores.csv')
assert len(S)==740*2*3 and len(I)==305*2*3
assert all(f in names for f in S.file.unique())
assert not S.duplicated(['file','seed','masked']).any()
assert not I.duplicated(['issue','seed','masked']).any()
assert (S.chars>0).all() and (I.chars>0).all()
assert S.year.between(1889,1894).all()
assert (S.extra_pieces_isig_minus_imvo == S.pieces_isig-S.pieces_imvo).all()

# Verify token scores by independent spm encoding of all sampled unmasked windows.
m={(seed,which):spm.SentencePieceProcessor(model_file=str(D/f'BPE_imvo{100 if which=="IMVO" else 0}_seed{seed}.model')) for seed in (27,28,29) for which in ('IMVO','ISIG')}
Wmap={w['file']:w for w in W}
if Wmap:
    for r in S[S.masked==0].itertuples():
        words=Wmap[r.file]['text']
        assert sum(x.isalpha() for x in words)==r.chars
        assert len(m[r.seed,'ISIG'].encode(words,out_type=int))==r.pieces_isig
        assert len(m[r.seed,'IMVO'].encode(words,out_type=int))==r.pieces_imvo

# Independent issue recomputation: exact sum of tokens/chars, not mean page E.
rec=S.groupby(['issue','year','seed','masked'],as_index=False).agg(chars=('chars','sum'),pi=('pieces_isig','sum'),pm=('pieces_imvo','sum'))
rec['e_isig']=100*rec.pi/rec.chars
rec['e_imvo']=100*rec.pm/rec.chars
rec['gain']=rec.e_isig-rec.e_imvo
J=I.merge(rec,on=['issue','year','seed','masked'],validate='one_to_one')
error=max(np.max(abs(J.E_isig-J.e_isig)),np.max(abs(J.E_imvo-J.e_imvo)),np.max(abs(J.advantage-J.gain)))
assert error<1e-12

# Verify primary aggregation and issue resampling confidence interval from independent arrays.
reported=pd.read_csv(D/'future_hypothesis_test.csv')
new=[]
for mask in (0,1):
    ii=I[I.masked==mask].groupby(['issue','year'],as_index=False).agg(a=('advantage','mean'))
    by={y:ii[ii.year==y].a.to_numpy() for y in range(1889,1895)}
    early=np.mean([by[y].mean() for y in (1889,1890,1891)])
    late=np.mean([by[y].mean() for y in (1892,1893,1894)])
    rng=np.random.default_rng(27127)
    # Main code has one RNG stream for mask=0 followed by mask=1.
    if mask==1:
        for _ in range(2000):
            for y in range(1889,1895):rng.choice(by[y],len(by[y]),replace=True)
    dist=[]
    for _ in range(2000):
        annual={y:np.mean(rng.choice(by[y],len(by[y]),replace=True)) for y in range(1889,1895)}
        dist.append(np.mean([annual[y] for y in (1892,1893,1894)])-np.mean([annual[y] for y in (1889,1890,1891)]))
    report=reported[reported.masked==mask].iloc[0]
    assert abs((late-early)-report.late_minus_early)<1e-12
    assert np.max(abs(np.quantile(dist,[.025,.975])-[report.ci_low,report.ci_high]))<1e-12
    new.append(dict(mask=mask,contrast=float(late-early)))

checks=dict(n_windows=int(S.file.nunique()),text_reencoding_verified=bool(Wmap),n_issues=len({w['issue'] for w in W}),n_tokenizer_rows=len(S),
            max_independent_issue_aggregation_error=float(error),
            future_years=sorted(set(w['year'] for w in W)),n_unknown_pieces=int(S[['isig_unknown','imvo_unknown']].to_numpy().sum()),
            bootstrap_contrasts=new,source_zip_sha256=audit['source_zip_sha256'],all_checks_passed=True)
(D/'FUTURE_VERIFICATION.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
