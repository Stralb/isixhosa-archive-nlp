#!/usr/bin/env python3
"""Retrospective diagnostic of cross-issue text overlap, with provenance.

The protocol is in PRE_AUDIT_PROTOCOL.md; input archive is not altered.
"""
import collections
import hashlib
import json
import re
import statistics
import zipfile
from pathlib import Path

ROOT=Path('/mnt/data/isiXhosa_publication_continuation')
ZIP=Path('/mnt/data/TxtLangFiles1870-1894_TB.zip')
RX_NAME=re.compile(r'^(IMVO|ISIG)_(1887|1888)-(\d\d)-(\d\d)_p(\d+)\.txt$')
RX_WORD=re.compile(r'[^\W\d_]+',re.UNICODE)
NVALUES=(8,12,20)

def digest(seq):
    return hashlib.blake2b(' '.join(seq).encode('utf-8'), digest_size=12).digest()

def tokens(data):
    return RX_WORD.findall(data.decode('utf-8','replace').casefold())

def load_pages():
    pages=[]
    with zipfile.ZipFile(ZIP) as z:
        for name in z.namelist():
            if '/XhTxt/' not in name or '/._' in name or not name.endswith('.txt'):continue
            fname=name.rsplit('/',1)[-1]; m=RX_NAME.fullmatch(fname)
            if not m: continue
            source,year,month,day,page=m.groups()
            split='train' if year=='1887' and int(month)<=9 else ('validation' if year=='1887' else 'old_test')
            pages.append({'name':fname,'issue':f'{source}_{year}-{month}-{day}', 'date':f'{year}-{month}-{day}', 'source':source,'split':split,'tokens':tokens(z.read(name))})
    return sorted(pages,key=lambda x:x['name'])

def index_ngrams(pages,n):
    # hashes represent exact ngrams, with deterministic fixed hash; source list allows cross-domain counts
    ix=collections.defaultdict(set)
    examples={}
    for p in pages:
        toks=p['tokens']
        for i in range(len(toks)-n+1):
            block=digest(toks[i:i+n]); ix[block].add((p['issue'],p['source']))
            if block not in examples: examples[block]=' '.join(toks[i:i+n])
    return ix,examples

def audit_eval(pages,index,n):
    page_out=[]
    for p in pages:
        toks=p['tokens']; k=len(toks)
        coverage=[False]*k; crosscov=[False]*k; in_source=[False]*k
        hits=0
        for i in range(k-n+1):
            h=digest(toks[i:i+n])
            matches=index.get(h,())
            matches=[(id,src) for id,src in matches if id!=p['issue']]
            if matches:
                hits+=1
                for j in range(i,i+n):coverage[j]=True
                if any(src!=p['source'] for _,src in matches):
                    for j in range(i,i+n):crosscov[j]=True
                if any(src==p['source'] for _,src in matches):
                    for j in range(i,i+n):in_source[j]=True
        page_out.append({'page':p['name'],'source':p['source'],'issue':p['issue'],'length':k,'matched_positions':sum(coverage),'matched_cross_positions':sum(crosscov),'matched_same_positions':sum(in_source),'matching_windows':hits})
    out={}
    for group in ['IMVO','ISIG','ALL']:
        records=[d for d in page_out if group=='ALL' or d['source']==group]
        n_words=sum(d['length'] for d in records)
        out[group]={'pages':len(records),'words':n_words,
            'covered_token_fraction':sum(d['matched_positions'] for d in records)/n_words if n_words else None,
            'cross_source_token_fraction':sum(d['matched_cross_positions'] for d in records)/n_words if n_words else None,
            'same_source_token_fraction':sum(d['matched_same_positions'] for d in records)/n_words if n_words else None,
            'pages_with_match':sum(d['matching_windows']>0 for d in records),
            'fraction_pages_with_match':sum(d['matching_windows']>0 for d in records)/len(records) if records else None,
            'total_matching_windows':sum(d['matching_windows'] for d in records)}
    return out,page_out

def main():
    pages=load_pages()
    assert len(pages)==594,(len(pages))
    train=[x for x in pages if x['split']=='train']
    valid=[x for x in pages if x['split']=='validation']
    test=[x for x in pages if x['split']=='old_test']
    report={'method':'Unicode alphabetic tokens, lowercased; within-page exact contiguous N-grams, cross-issue only',
            'input_archive':ZIP.name,'input_sha256':hashlib.sha256(ZIP.read_bytes()).hexdigest(),
            'pages_by_split':{s:len(q) for s,q in [('train',train),('validation',valid),('old_test',test)]},
            'issues_by_split':{s:len({x['issue'] for x in q}) for s,q in [('train',train),('validation',valid),('old_test',test)]},
            'source_counts':{s:sum(p['source']==s for p in pages) for s in ['IMVO','ISIG']},'ngram_results':{}}
    for n in NVALUES:
        idx,examples=index_ngrams(train,n)
        heldout_result, page_stats=audit_eval(test,idx,n)
        val_result,_=audit_eval(valid,idx,n)
        report['ngram_results'][str(n)]={'train_shingle_unique':len(idx),'train_to_1888':heldout_result,'train_to_1887_val':val_result}
        if n==12:
            # inspect first pairs with strongest direct overlap, but avoid material from 1886
            ranked=sorted(page_stats,key=lambda p:(p['matched_positions'],p['matching_windows']),reverse=True)
            report['top_matched_test_pages']=[{'page':p['page'],'matched_positions':p['matched_positions'],'length':p['length'],'matching_windows':p['matching_windows']} for p in ranked[:12]]
            examples_out=[]
            seen=set()
            for p in test:
                ts=p['tokens']
                for i in range(len(ts)-n+1):
                    sh=digest(ts[i:i+n]); seen_sources=idx.get(sh)
                    if not seen_sources or sh in seen:continue
                    seen.add(sh)
                    examples_out.append({'test_page':p['name'],'match_training_issue_count':len(seen_sources),'cross_source':any(src!=p['source'] for _,src in seen_sources),'text':' '.join(ts[i:i+n])})
            examples_out.sort(key=lambda x:(x['cross_source'],x['match_training_issue_count']),reverse=True)
            report['example_phrases_12gram']=examples_out[:18]
        print(f'Finished n={n}: test covered {heldout_result["ALL"]["covered_token_fraction"]:.4%}',flush=True)
    (ROOT/'overlap_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({'pages_by_split':report['pages_by_split'], 'issues_by_split':report['issues_by_split'], 'ngram_results':report['ngram_results']},indent=2))

if __name__=='__main__':main()
