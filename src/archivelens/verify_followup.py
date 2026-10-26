#!/usr/bin/env python3
"""Independent replay of follow-up permutation p-values, issue influence and repeat masks.

No imports from followup.py or audit.py: separate raw-ZIP parser, aggregators
and full 4999-draw randomization logic. Keeps 1887 candidate features fixed.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,re
from collections import Counter,defaultdict
from pathlib import Path
from zipfile import ZipFile
import numpy as np

FNAME=re.compile(r'/XhTxt/(IMVO|ISIG)_(1885|1888)-(\d\d)-(\d\d)_p(\d\d\d)\.txt$')
TOKEN=re.compile(r'[a-z]+')
SOURCES=('IMVO','ISIG')
METHODS=('word_mass','page_equal','issue_equal','month_equal')
RNG=20261027
N=4999
EXPECTED='375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readcsv(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def source_pages(data, features):
    allrows=defaultdict(list)
    with ZipFile(data) as z:
        for path in sorted(z.namelist()):
            m=FNAME.search(path)
            if not m:continue
            src,year,month,day,position=m.groups()
            words=TOKEN.findall(z.read(path).decode('utf8','replace').lower())
            if len(words)<40:continue
            denom=sum(max(0,len(w)-3) for w in words)
            if denom==0:continue
            four=Counter(w[i:i+4] for w in words for i in range(len(w)-3))
            rr={'source':src,'year':int(year),'month':int(month),'issue':f'{src}_{year}-{month}-{day}',
                'page':int(position),'words':words,'num_words':len(words),'den':denom,
                'counts':np.array([four[pat] for pat in features],dtype=np.float64)}
            allrows[(src,int(year))].append(rr)
    return allrows


def group_issues(pages):
    byissue=defaultdict(list)
    for p in pages:byissue[p['issue']].append(p)
    result=[]
    for issue,group in sorted(byissue.items()):
        r={'issue':issue,'year':group[0]['year'],'month':group[0]['month'],'pages':len(group),
           'char_mass':sum(p['den'] for p in group),
           'counts':sum((p['counts'] for p in group),np.zeros(len(group[0]['counts']))),
           'mean_page_rate':sum((p['counts']/p['den'] for p in group),np.zeros(len(group[0]['counts'])))}
        assert all(z['month']==r['month'] for z in group)
        result.append(r)
    return result


def rate_est(issues):
    a=issues
    count=np.array([r['counts'] for r in a]);den=np.array([r['char_mass'] for r in a])
    ps=np.array([r['mean_page_rate'] for r in a]);n=np.array([r['pages'] for r in a])
    r=count/den[:,None]
    bymonth=defaultdict(list)
    for j,g in enumerate(a):bymonth[g['month']].append(j)
    month=np.array([r[bymonth[m]].mean(axis=0) for m in sorted(bymonth)]).mean(axis=0)
    if len(bymonth)!=12:month[:]=np.nan
    arr=np.array([count.sum(axis=0)/den.sum(),ps.sum(axis=0)/n.sum(),r.mean(axis=0),month])*10000
    return arr


def standalone_adjust(p):
    order=np.argsort(p,kind='stable');sorted_p=np.array(p)[order];nn=len(p)
    bh=np.minimum.accumulate((sorted_p*nn/np.arange(1,nn+1))[::-1])[::-1]
    holm=np.maximum.accumulate(sorted_p*np.arange(nn,0,-1))
    res=[]
    for x in (bh,holm):
        y=np.empty(nn);y[order]=np.minimum(1,np.maximum(0,x));res.append(y)
    return res


def replay_permutation(src,allrows,calibration):
    a=group_issues(allrows[(src,1885)]);b=group_issues(allrows[(src,1888)])
    observed=rate_est(b)-rate_est(a)
    stored=calibration[src]
    for j,r in enumerate(stored):
        if abs(observed[0,j]-float(r['temporal_mass_delta']))>1.1e-8:
            raise AssertionError('Original delta differs in independent verification')
    gap=np.max(abs(observed[1:]-observed[0][None,:]),axis=0)
    obs_count=int(np.any(observed[0][None,:]*observed[1:]<0,axis=0).sum())
    combined=a+b
    month_indices={month:( [i for i,g in enumerate(a) if g['month']==month],
                           [len(a)+i for i,g in enumerate(b) if g['month']==month]) for month in range(1,13)}
    g=np.random.default_rng(RNG+SOURCES.index(src))
    larger1=np.zeros(120,dtype=int);larger2=np.zeros(120,dtype=int)
    counts=[]
    for _ in range(N):
        ia=[];ib=[]
        for month in range(1,13):
            olds,news=month_indices[month]
            ix=g.permutation(np.asarray(olds+news))
            ia.extend(ix[:len(olds)])
            ib.extend(ix[len(olds):])
        current=rate_est([combined[j] for j in ib])-rate_est([combined[j] for j in ia])
        larger1 += abs(current[0])>=abs(observed[0])-1e-12
        larger2 += np.max(abs(current[1:]-current[0][None,:]),axis=0)>=gap-1e-12
        counts.append(int(np.any(current[0][None,:]*current[1:]<0,axis=0).sum()))
    p1=(larger1+1)/5000;p2=(larger2+1)/5000
    b1,h1=standalone_adjust(p1);b2,h2=standalone_adjust(p2)
    for j,r in enumerate(stored):
        for col,val in [('max_estimand_gap',gap[j]),('temporal_perm_p',p1[j]),('estimand_gap_perm_p',p2[j]),
                        ('temporal_bh_q',b1[j]),('temporal_holm_p',h1[j]),
                        ('estimand_gap_bh_q',b2[j]),('estimand_gap_holm_p',h2[j])]:
            if abs(float(r[col])-float(val))>1e-9:
                raise AssertionError(f'Independent permutation mismatch {src} {r["pattern"]} {col}: {r[col]} vs {val}')
    a=np.asarray(counts)
    out={'observed':obs_count,'null_mean':float(a.mean()),
         'p_upper_tail':(1+int((a>=obs_count).sum()))/5000,
         'p_lower_tail':(1+int((a<=obs_count).sum()))/5000}
    print('PASS',src,'independent full 4,999-permutation replay','max gap',float(gap.max()),out)
    return out


def verify_influences(allrows,top):
    checked=0;maxerr=0
    for r in top:
        src=r['source'];pat=r['pattern'];j=PAT_INDEX[pat]
        years={yr:allrows[(src,yr)] for yr in (1885,1888)}
        all0=rate_est(group_issues(years[1885]))
        all1=rate_est(group_issues(years[1888]))
        original=all1-all0
        left=[p for p in years[int(r['year'])] if p['issue']!=r['issue']]
        if not left:raise AssertionError('LOO no pages')
        other=rate_est(group_issues(left))
        new=(all1-other) if int(r['year'])==1885 else (other-all0)
        for key,computed in [('mass_effect_change',new[0,j]-original[0,j]),
                             ('mass_effect_without_issue',new[0,j]),
                             ('page_effect_change',new[1,j]-original[1,j])]:
            err=abs(float(r[key])-computed);maxerr=max(maxerr,err)
            if err>1.1e-8:raise AssertionError(f'Independent LOO mismatch {src} {pat} {r["issue"]} {key}: {err}')
        checked+=1
    print('PASS',checked,'independent leave-one-issue-out top-issue effect checks, max error',maxerr)


def verify_duplicates(allrows,cover,maskrows):
    checked=0
    bymask={(r['source'],int(r['year']),r['issue'],int(r['page_no'])):r for r in maskrows}
    bycover={(r['source'],int(r['year'])):r for r in cover}
    for src in SOURCES:
        pages=allrows[(src,1885)]+allrows[(src,1888)]
        first={};repeated=set()
        # Independent exact word tuple dictionary: no hashing or main-library routines.
        for p in pages:
            w=p['words']
            for i in range(len(w)-12):
                sh=tuple(w[i:i+13]);previous=first.get(sh)
                if previous is None:first[sh]=p['issue']
                elif previous!=p['issue']:repeated.add(sh)
        if int(bycover[(src,1885)]['exact_repeated_13gram_types_in_source'])!=len(repeated):
            raise AssertionError(f'Source {src} repeated-sequence type count differs')
        sums=Counter()
        for p in pages:
            w=p['words']; mask=np.zeros(len(w)+1,dtype=np.int32)
            for i in range(len(w)-12):
                if tuple(w[i:i+13]) in repeated:
                    mask[i]+=1;mask[i+13]-=1
            covered=int((np.cumsum(mask[:-1])>0).sum())
            k=(src,p['year'],p['issue'],p['page'])
            if k not in bymask:raise AssertionError(f'Missing duplicate-mask row {k}')
            if int(bymask[k]['masked_words'])!=covered:
                raise AssertionError(f'Exact tuple vs digest mask disagree {k} expected {covered}')
            sums[(p['year'],'masked')]+=covered
            sums[(p['year'],'total')]+=len(w)
            checked+=1
        for year in (1885,1888):
            c=bycover[(src,year)]
            assert int(c['masked_words'])==sums[(year,'masked')]
            assert int(c['original_words'])==sums[(year,'total')]
        print('PASS',src,'independent exact-tuple repeated phrase mask',len(repeated),'repeated shingles')
    print('PASS',checked,'independent page-level exact repeated-word masked counts')


PAT_INDEX={}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--root',type=Path,default=Path('.'))
    args=ap.parse_args();root=args.root
    if sha(args.data)!=EXPECTED:raise AssertionError('ZIP changed')
    base=root/'results/archivelens/followup'
    candidates=readcsv(root/'results/archivelens/candidate_1887.csv')
    names=[x['pattern'] for x in candidates]
    if len(names)!=120:raise AssertionError('Candidate list changed')
    global PAT_INDEX;PAT_INDEX={pat:i for i,pat in enumerate(names)}
    pages=source_pages(args.data,names)
    c=readcsv(base/'calibration.csv'); bysrc=defaultdict(list)
    for row in c:bysrc[row['source']].append(row)
    for src in SOURCES:replay_permutation(src,pages,bysrc)
    verify_influences(pages,readcsv(base/'influential_issues.csv'))
    verify_duplicates(pages,readcsv(base/'duplicate_coverage.csv'),readcsv(base/'page_duplicate_masks.csv'))
    print('PASS: independent archive reread, all 4,999 permutations per title, all LOO witnesses, all 13-word masks')

if __name__=='__main__':main()
