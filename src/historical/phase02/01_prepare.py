#!/usr/bin/env python3
"""Prepare leakage-audited issue splits and reproducible original-archive metadata.
No synthetic periods, masks, or benchmark outcomes generated here.
"""
from pathlib import Path
import argparse, csv, hashlib, json, re, zipfile, collections
import numpy as np, pandas as pd

ORIGIN_SHA='375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5'
WORD_RX=re.compile(r'[^\W\d_]+', re.UNICODE)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(2**20),b''):h.update(part)
    return h.hexdigest()

def generate(origin, sourcezip, out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    assert sha(sourcezip)==ORIGIN_SHA,'Wrong original ZIP'
    p1=Path(origin)
    pages=pd.read_csv(p1/'pages.csv')
    counts=np.load(p1/'counts.npz',allow_pickle=False)
    original=np.asarray(counts['word_counts'],dtype=np.int32)
    assert original.shape==(292,200)
    assert np.array_equal(counts['word_totals'],pages.n_tokens.values)
    words=pd.read_csv(p1/'features.csv').feature.tolist()
    assert len(words)==len(set(words))==200
    split_lookup={}
    for source in ['IMVO','ISIG']:
        issue_list=sorted(pages.loc[pages.source==source,'issue'].unique(),key=lambda issue: hashlib.sha256(('P2-SPLIT-20261024|'+issue).encode()).hexdigest())
        n=len(issue_list); end1=int(n*.60);end2=end1+int(n*.20)
        for issue in issue_list[:end1]:split_lookup[issue]='train'
        for issue in issue_list[end1:end2]:split_lookup[issue]='dev'
        for issue in issue_list[end2:]:split_lookup[issue]='test'
    pages['split']=pages.issue.map(split_lookup)
    assert pages.split.notna().all()
    for src,group in pages.groupby('source'):
        print('SPLIT',src,group.groupby('split').issue.nunique().to_dict(),group.groupby('split').size().to_dict())
    trainmask=pages.split=='train'
    rank=sorted(range(200),key=lambda j:(-int(original[trainmask,j].sum()),words[j]))
    selected=rank[:80]
    feat=[words[j] for j in selected]
    selectcounts=original[:,selected]
    train_sum=selectcounts[trainmask].sum(axis=0)
    idx=np.arange(80)
    sig_types=np.array(['NULL']*80)
    sig_types[[18,24,30,36,42,48]]='UP'
    sig_types[[6,8,10,12,14,16]]='DOWN'
    ft=pd.DataFrame({'feature_index':idx,'word_form':feat,'training_count':train_sum,'phase1_pool_index':selected,'signal':sig_types})
    ft.to_csv(out/'feature_manifest.csv',index=False)
    assert (selectcounts.sum(axis=1)<=pages.n_tokens.values).all()
    # Exact 13-token cross-ISSUE repetition flags (form overlap, NOT confirmed reprints).
    seen={};gramsets=[]
    with zipfile.ZipFile(sourcezip) as z:
        for rec in pages.itertuples():
            original_bytes=z.read(rec.zip_entry)
            assert hashlib.sha256(original_bytes).hexdigest()==rec.text_sha256
            txt=original_bytes.decode('utf-8','replace').casefold()
            tokens=WORD_RX.findall(txt)
            assert len(tokens)==rec.n_tokens
            # Check feature totals independently from original raw text.
            counter=collections.Counter(tokens)
            for j,w in enumerate(feat):
                assert counter[w]==int(selectcounts[rec.Index,j]), (rec.issue,w)
            gs=set(tuple(tokens[i:i+13]) for i in range(max(0,len(tokens)-12)))
            gramsets.append(gs)
            for g in gs:
                if g not in seen:seen[g]=rec.issue
    repeated=set()
    for i,rec in pages.iterrows():
        for g in gramsets[i]:
            if seen[g]!=rec.issue:repeated.add(g)
    cnt=np.asarray([len(x & repeated) for x in gramsets],dtype=int)
    pages['shared_13gram_count']=cnt
    pages['has_cross_issue_13gram']=cnt>0
    pages['month']=pages.date.str[5:7].astype(int)
    y=selectcounts[:,:4].sum(axis=1)/pages.n_tokens.values
    avg=float(y[trainmask].mean());std=float(y[trainmask].std(ddof=0))
    assert std>0
    pages['hidden_text_proxy']=np.clip((y-avg)/std,-2.5,2.5)
    pages.to_csv(out/'frame.csv',index=False)
    np.savez_compressed(out/'counts.npz',counts=selectcounts,total_tokens=pages.n_tokens.values)
    print('REPRINT FLAG',int(pages.has_cross_issue_13gram.sum()),'split',pages.groupby('split').has_cross_issue_13gram.sum().to_dict())
    print('FEATURES',len(feat),'NULL',(sig_types=='NULL').sum(),'UP',(sig_types=='UP').sum(),'DOWN',(sig_types=='DOWN').sum())
    obj={'schema':'PRHI-P2-1885-audit-frame-v1','source_sha256':ORIGIN_SHA,
         'phase1_counts_sha256':sha(p1/'counts.npz'),'phase1_pages_sha256':sha(p1/'pages.csv'),
         'split_issue_counts':{src:{sp:int(pages.loc[(pages.source==src)&(pages.split==sp),'issue'].nunique()) for sp in ['train','dev','test']} for src in ['IMVO','ISIG']},
         'features':len(feat),'selected_train_only_from_known_phase1_200_pool':True,
         'signal_up_feature_indices':[18,24,30,36,42,48],'signal_down_feature_indices':[6,8,10,12,14,16],
         'hidden_proxy_train_mean':avg,'hidden_proxy_train_std':std,
         'repeated_13gram_pages':int((cnt>0).sum()),
         'data_caveat':'Phase1 original 200-feature vocabulary selected using the same 1885 archive; even TEST not project-level independent.'}
    (out/'frame_manifest.json').write_text(json.dumps(obj,indent=2)+'\n')
    return obj

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--p1',required=True);ap.add_argument('--zip',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();generate(a.p1,a.zip,a.out)
