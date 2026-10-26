#!/usr/bin/env python3
"""Independent raw-archive verification of candidate ranking and 960 year contrasts.

Does not import or call ArchiveLens feature extraction/aggregation routines.
"""
import argparse,csv,re
from collections import Counter,defaultdict
from pathlib import Path
from zipfile import ZipFile

NAME=re.compile(r"/XhTxt/(IMVO|ISIG)_(1885|1887|1888)-(\d\d)-(\d\d)_p(\d{3})\.txt$")
WORD=re.compile(r"[a-z]+")

def extract(path,patterns):
    out=defaultdict(list)
    train={s:Counter() for s in ("IMVO","ISIG")}
    positions=Counter()
    with ZipFile(path) as z:
        for name in sorted(z.namelist()):
            if name.startswith("__MACOSX") or "/._" in name:continue
            m=NAME.search(name)
            if not m:continue
            src,year,month,day,page=m.groups()
            words=WORD.findall(z.read(name).decode("utf-8").lower())
            if len(words)<40:continue
            den=sum(max(0,len(w)-3) for w in words)
            if den==0:continue
            counts=Counter(w[k:k+4] for w in words for k in range(len(w)-3))
            if year=="1887":
                train[src].update(counts);positions[src]+=den
            else:
                out[(src,int(year))].append({
                    "month":int(month),"issue":f"{src}_{year}-{month}-{day}",
                    "den":den,"counts":{p:counts[p] for p in patterns}})
    names=sorted(set(train["IMVO"])|set(train["ISIG"]),
                 key=lambda pat:(-.5*(train["IMVO"][pat]/positions["IMVO"]+
                                     train["ISIG"][pat]/positions["ISIG"]),pat))
    return out,names[:len(patterns)]

def reference_rates(pages,pattern):
    mass=10000*sum(p["counts"][pattern] for p in pages)/sum(p["den"] for p in pages)
    page=10000*sum(p["counts"][pattern]/p["den"] for p in pages)/len(pages)
    issues=defaultdict(list)
    for p in pages:issues[(p["month"],p["issue"])].append(p)
    rates_by_month=defaultdict(list)
    for (month,issue),group in issues.items():
        rate=sum(p["counts"][pattern] for p in group)/sum(p["den"] for p in group)
        rates_by_month[month].append(rate)
    issue=10000*sum(sum(v) for v in rates_by_month.values())/sum(len(v) for v in rates_by_month.values())
    month=10000*sum(sum(v)/len(v) for v in rates_by_month.values())/len(rates_by_month)
    return dict(zip(("word_mass","page_equal","issue_equal","month_equal"),(mass,page,issue,month)))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("results/archivelens"))
    args=p.parse_args()
    with (args.out/"candidate_1887.csv").open() as f:
        candidates=[r["pattern"] for r in csv.DictReader(f)]
    with (args.out/"estimands.csv").open() as f:
        saved=list(csv.DictReader(f))
    source,ranked=extract(args.data,candidates)
    if ranked!=candidates:raise AssertionError("Independent 1887 candidate selection does not match")
    by={}
    for src in ("IMVO","ISIG"):
        for year in (1885,1888):
            for gram in candidates:
                by[(src,year,gram)]=reference_rates(source[(src,year)],gram)
    max_error=0
    for r in saved:
        src,pat,est=r["source"],r["pattern"],r["estimand"]
        a=by[(src,1885,pat)][est];b=by[(src,1888,pat)][est]
        for col,v in (("rate_1885",a),("rate_1888",b),("delta",b-a)):
            err=abs(float(r[col])-v);max_error=max(err,max_error)
            if err>1.1e-8:raise AssertionError(f"Failed {src}/{pat}/{est}/{col}: {err}")
    if len(saved)!=960:raise AssertionError("Expected 960 estimate contrasts")
    print(f"PASS: independent candidate ranking and 960 temporal effects x 3 fields; max abs error {max_error:.3g}")

if __name__=="__main__":main()
