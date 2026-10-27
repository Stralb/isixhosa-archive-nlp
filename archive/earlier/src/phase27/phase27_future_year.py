#!/usr/bin/env python3
"""Phase 27B prospective chronologically later Imvo-only tokenizer drift test."""
import argparse,collections,csv,hashlib,json,re,zipfile
from pathlib import Path
import numpy as np
import sentencepiece as spm

SEED=27127
SEEDS=(27,28,29)
YEARS=set(range(1889,1895))
NAME=re.compile(r'IMVO_(\d{4})-(\d{2})-(\d{2})_p\d{3}\.txt$')
ALPHA=re.compile(r'[^\W\d_]+',re.UNICODE)
BAD={'imvo','zabantsundu','isigidimi','samaxosa'}
SUBSTR=('imvo','isigidi','samaxosa','zabantsundu','bantsundu','ntsundu','mnyama','xosa','jabavu','gqoba','makiwane')
EXACT={'mr','sir','lovedale','b','g','e','th'}

def count_alpha(t):return sum(ch.isalpha() for ch in t)
def dump_csv(path,rows):
    assert rows
    with open(path,'w',newline='',encoding='utf8') as f:
        dw=csv.DictWriter(f,fieldnames=list(rows[0]));dw.writeheader();dw.writerows(rows)

def data_build(zipfile_path,out):
    windows=[]
    total=collections.Counter()
    with zipfile.ZipFile(zipfile_path) as z:
        for f in sorted(z.namelist()):
            if '/XhTxt/' not in f or not f.endswith('.txt') or '__MACOSX' in f:continue
            name=f.rsplit('/',1)[-1]
            mt=NAME.fullmatch(name)
            if not mt:continue
            year,month,day=map(int,mt.groups())
            if year not in YEARS:continue
            total[year]+=1
            txt=z.read(f).decode('utf-8-sig','replace').replace('\r\n','\n')
            choices=[]
            for block in re.split(r'\n\s*\n',txt):
                if len(block.split())<85:continue
                words=[w for w in ALPHA.findall(block.lower()) if w not in BAD]
                if len(words)>=90:choices.append(words)
            if not choices:continue
            chosen=choices[len(choices)//2]
            start=(len(chosen)-90)//2
            tok=chosen[start:start+90]
            assert len(tok)==90
            windows.append(dict(file=name,issue=f'IMVO_{year}-{month:02d}-{day:02d}',
                                year=year,month=month,text=' '.join(tok),tokens=tok))
    duplicate_counts=collections.Counter(w['text'] for w in windows)
    dup=sum(duplicate_counts[w['text']]>1 for w in windows)
    windows=[w for w in windows if duplicate_counts[w['text']]==1]
    assert windows
    with open(out/'future_1889_1894_windows.jsonl','w',encoding='utf8') as f:
        for w in windows:f.write(json.dumps(w,ensure_ascii=False)+'\n')
    audit=dict(source_zip_sha256=hashlib.sha256(Path(zipfile_path).read_bytes()).hexdigest(),
               original_pages_per_year=dict(total),
               windows_per_year={y:sum(w['year']==y for w in windows) for y in sorted(YEARS)},
               issues_per_year={y:len({w['issue'] for w in windows if w['year']==y}) for y in sorted(YEARS)},
               excluded_duplicate_window_occurrences=dup,
               window_file_sha256=hashlib.sha256((out/'future_1889_1894_windows.jsonl').read_bytes()).hexdigest())
    (out/'future_sampling_audit.json').write_text(json.dumps(audit,indent=2))
    return windows,audit

def score(windows,out):
    models={(seed,source):spm.SentencePieceProcessor(model_file=str(out/f'BPE_imvo{100 if source=="IMVO" else 0}_seed{seed}.model'))
            for seed in SEEDS for source in ['IMVO','ISIG']}
    rows=[];unknown=[]
    for w in windows:
        for masked in (0,1):
            txt=w['text'] if not masked else ' '.join(t for t in w['tokens']
                 if t not in EXACT and not any(x in t for x in SUBSTR))
            chars=count_alpha(txt)
            if chars<=0:raise RuntimeError('empty masked window '+w['file'])
            for seed in SEEDS:
                a=models[seed,'ISIG'].encode(txt,out_type=int)
                b=models[seed,'IMVO'].encode(txt,out_type=int)
                rows.append(dict(file=w['file'],issue=w['issue'],year=w['year'],month=w['month'],
                                 seed=seed,masked=masked,chars=chars,
                                 pieces_isig=len(a),pieces_imvo=len(b),
                                 extra_pieces_isig_minus_imvo=len(a)-len(b),
                                 isig_unknown=sum(v==0 for v in a),
                                 imvo_unknown=sum(v==0 for v in b)))
    dump_csv(out/'future_page_scores.csv',rows)
    return rows

def analyze(rows,out):
    groups=collections.defaultdict(list)
    for r in rows:
        groups[r['issue'],r['seed'],r['masked']].append(r)
    issue_rows=[]
    for (issue,seed,masked),ss in sorted(groups.items()):
        chars=sum(x['chars'] for x in ss)
        ep=100*sum(x['pieces_isig'] for x in ss)/chars
        ip=100*sum(x['pieces_imvo'] for x in ss)/chars
        issue_rows.append(dict(issue=issue,year=ss[0]['year'],month=ss[0]['month'],
                               seed=seed,masked=masked,n_pages=len(ss),chars=chars,
                               E_isig=ep,E_imvo=ip,advantage=ep-ip))
    dump_csv(out/'future_issue_scores.csv',issue_rows)
    yearly=[];summary=[]
    rng=np.random.default_rng(SEED)
    for masked in (0,1):
        # first average all tokenizer seeds for each issue
        iss=collections.defaultdict(list)
        for r in issue_rows:
            if r['masked']==masked:iss[r['issue']].append(r)
        byyear=collections.defaultdict(list)
        for key,vs in iss.items():
            assert len(vs)==3
            year=vs[0]['year']
            byyear[year].append(float(np.mean([x['advantage'] for x in vs])))
        assert all(len(byyear[y])>0 for y in YEARS)
        for year in sorted(YEARS):
            rr=[v for v in issue_rows if v['masked']==masked and v['year']==year]
            for seed in SEEDS:
                r=[v for v in rr if v['seed']==seed]
                yearly.append(dict(masked=masked,year=year,seed=seed,n_issues=len(r),
                                   avg_E_isig=float(np.mean([v['E_isig'] for v in r])),
                                   avg_E_imvo=float(np.mean([v['E_imvo'] for v in r])),
                                   avg_advantage=float(np.mean([v['advantage'] for v in r]))))
        early=float(np.mean([np.mean(byyear[y]) for y in (1889,1890,1891)]))
        late=float(np.mean([np.mean(byyear[y]) for y in (1892,1893,1894)]))
        diff=late-early
        bs=[]
        for _ in range(2000):
            annual={y:float(np.mean(rng.choice(byyear[y],len(byyear[y]),replace=True))) for y in sorted(YEARS)}
            bs.append(float(np.mean([annual[y] for y in (1892,1893,1894)])-
                            np.mean([annual[y] for y in (1889,1890,1891)])))
        lo,hi=np.quantile(bs,[.025,.975])
        summary.append(dict(masked=masked,early_1889_1891=early,late_1892_1894=late,
                            late_minus_early=diff,ci_low=float(lo),ci_high=float(hi),
                            criterion_pass=int(diff>=.30 and lo>0),n_issues=sum(len(byyear[y]) for y in YEARS)))
    dump_csv(out/'future_yearly_scores.csv',yearly)
    dump_csv(out/'future_hypothesis_test.csv',summary)
    print('FUTURE_TEST',json.dumps(summary,indent=2),flush=True)
    return yearly,summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-zip',default='/mnt/data/TxtLangFiles1870-1894_TB.zip')
    ap.add_argument('--output',default=str(Path(__file__).parent))
    args=ap.parse_args();out=Path(args.output)
    windows,audit=data_build(args.data_zip,out)
    print('SAMPLING',json.dumps(audit),flush=True)
    rows=score(windows,out)
    yearly,summary=analyze(rows,out)
    (out/'future_results.json').write_text(json.dumps(dict(audit=audit,summary=summary,yearly=yearly),indent=2))
if __name__=='__main__':main()
