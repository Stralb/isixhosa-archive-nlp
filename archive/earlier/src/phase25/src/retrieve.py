#!/usr/bin/env python3
"""Deterministic historical newspaper retrieval baselines, without qrels or tuning.

Inputs: original handoff's pages.jsonl and an independently authored queries.csv.
Outputs: ranked runs.csv, run_manifest.json, corpus_hashes.csv.
The query list does NOT need relevance labels; outputs must be blindly judged.
"""
from __future__ import annotations
import argparse,csv,json,hashlib,math,re,sys,time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

TOKEN_PATTERN=r'(?u)\b[^\W\d_]+\b'
WORD_RE=re.compile(TOKEN_PATTERN)
SOURCES={'IMVO','ISIG'}


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for buf in iter(lambda:f.read(1048576),b''):h.update(buf)
    return h.hexdigest()


def read_queries(p, diagnostic=False):
    with Path(p).open(encoding='utf-8-sig',newline='') as f: rr=list(csv.DictReader(f))
    if not rr:raise ValueError('empty query set')
    ids=set()
    for r in rr:
        if not {'query_id','query_text','intent','source_scope','author_id','topic_id'}.issubset(r):raise ValueError('missing query schema')
        k=r['query_id'].strip()
        if not k or k in ids or not r['query_text'].strip():raise ValueError('empty or duplicate query')
        ids.add(k)
        if r['intent'] not in {'evidence','provenance','verbatim_reuse'}:raise ValueError('unknown intent')
        if not diagnostic and r.get('independent_author_attestation')!='YES':raise ValueError('human attestation absent; --diagnostic only for sandbox testing')
    return rr


def read_pages(path):
    arr=[json.loads(s) for s in Path(path).open(encoding='utf-8')]
    if len(set(d['id'] for d in arr))!=len(arr):raise ValueError('duplicate pages')
    return arr


def chunks_from_pages(pages, chunk_tokens=180, overlap=45):
    if chunk_tokens<=overlap or overlap<0:raise ValueError('overlap must be less than chunk size')
    chunks=[];owners=[];chunk_ids=[]
    stride=chunk_tokens-overlap
    for doc_idx,d in enumerate(pages):
        tokens=WORD_RE.findall(d['raw'])
        if not tokens:continue
        for start in range(0,len(tokens),stride):
            seg=tokens[start:start+chunk_tokens]
            if len(seg)<max(20,chunk_tokens//4) and start>0:break
            chunks.append(' '.join(seg));owners.append(doc_idx)
            chunk_ids.append((d['id'],start))
            if start+chunk_tokens>=len(tokens):break
    return chunks,np.asarray(owners,dtype=np.int32),chunk_ids


class SparseBM25:
    def __init__(self,texts):
        self.cv=CountVectorizer(lowercase=True,token_pattern=TOKEN_PATTERN,dtype=np.float32)
        x=self.cv.fit_transform(texts).tocsr()
        n=x.shape[0]
        lengths=np.asarray(x.sum(axis=1)).ravel().astype(np.float32)
        df=np.asarray((x>0).sum(axis=0)).ravel()
        idf=np.log1p((n-df+.5)/(df+.5))
        avg=max(float(lengths.mean()),1.)
        coo=x.tocoo()
        weights=coo.data*2.2/(coo.data+1.2*(.25+.75*lengths[coo.row]/avg))*idf[coo.col]
        self.matrix=sp.csr_matrix((weights,(coo.row,coo.col)),shape=coo.shape,dtype=np.float32)
        self.n_docs=n
    def rank_one(self,query):
        q=self.cv.transform([query]);q.data[:]=1
        scores=np.asarray((q@self.matrix.T).toarray()).ravel()
        return scores


class SparseTFIDF:
    def __init__(self,texts):
        self.cv=TfidfVectorizer(lowercase=True,token_pattern=TOKEN_PATTERN,sublinear_tf=True,dtype=np.float32,norm='l2')
        self.matrix=self.cv.fit_transform(texts).tocsr()
    def rank_one(self,query):
        q=self.cv.transform([query]);return np.asarray((q@self.matrix.T).toarray()).ravel()


def index_order(scores,limit, positive_only=False):
    a=np.asarray(scores)
    # deterministic rank: high scores first; identical scores by page input order
    candidate=np.flatnonzero(a>0) if positive_only else np.arange(len(a))
    return sorted(candidate,key=lambda j:(-float(a[j]),int(j)))[:limit]


def fuse_rrf(a,b,limit,k=60):
    ranks={}
    for indices in (a,b):
        for pos,j in enumerate(indices,1):ranks[j]=ranks.get(j,0)+1/(k+pos)
    return sorted(ranks,key=lambda j:(-ranks[j],j))[:limit],ranks


def read_data(args):
    queries=read_queries(args.queries,diagnostic=args.diagnostic)
    pages=read_pages(args.pages)
    if not pages:raise ValueError('no pages')
    return pages,queries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pages',required=True)
    parser.add_argument('--queries',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--topk',type=int,default=20)
    parser.add_argument('--chunk-tokens',type=int,default=180)
    parser.add_argument('--chunk-overlap',type=int,default=45)
    parser.add_argument('--diagnostic',action='store_true')
    a=parser.parse_args()
    if a.topk<10:raise SystemExit('topk must be >=10 for the prospective judging pool')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    t=time.time();pages,queries=read_data(a)
    page_text=[d['raw'] for d in pages]
    print('Loaded',len(pages),'pages,',len(queries),'queries',flush=True)
    page_bm=SparseBM25(page_text)
    page_tf=SparseTFIDF(page_text)
    chunks,owners,chunkids=chunks_from_pages(pages,a.chunk_tokens,a.chunk_overlap)
    chunk_bm=SparseBM25(chunks)
    print('Indexed',len(chunks),'passages in',round(time.time()-t,1),'seconds',flush=True)
    heads=['system','query_id','doc_id','rank','score']
    with (out/'runs.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,heads);w.writeheader()
        for i,q in enumerate(queries,1):
            text=q['query_text']
            bm=page_bm.rank_one(text)
            tf=page_tf.rank_one(text)
            ch=chunk_bm.rank_one(text)
            aggregate=np.zeros(len(pages),dtype=np.float32)
            np.maximum.at(aggregate,owners,ch)
            top_bm=index_order(bm,100)
            top_chunk=index_order(aggregate,100)
            fusion, f_scores=fuse_rrf(top_bm,top_chunk,a.topk)
            date=q.get('requested_issue_date','').strip()
            src=q.get('source_scope','')
            if q['intent']=='provenance' and date:
                matching=[j for j,d in enumerate(pages) if d.get('date')==date and (src not in {'IMVO','ISIG'} or d.get('source')==src)]
                ranked_all=index_order(bm,len(pages))
                ondate=sorted(matching,key=lambda j:(-float(bm[j]),j))
                matching_set=set(matching)
                metarank=(ondate+[j for j in ranked_all if j not in matching_set])[:a.topk]
            else:
                metarank=index_order(bm,a.topk)
            out_indices={
                'BM25_PAGE':index_order(bm,a.topk),
                'BM25_METADATA_DATE':metarank,
                'BM25_PASSAGE_MAX':index_order(aggregate,a.topk),
                'TFIDF_PAGE':index_order(tf,a.topk),
                'RRF_PAGE_PASSAGE_60':fusion}
            vals={'BM25_PAGE':bm,'BM25_METADATA_DATE':bm,'BM25_PASSAGE_MAX':aggregate,'TFIDF_PAGE':tf}
            for name,ordered in out_indices.items():
                for rank,j in enumerate(ordered,1):
                    score=f_scores[j] if name=='RRF_PAGE_PASSAGE_60' else float(vals[name][j])
                    w.writerow({'system':name,'query_id':q['query_id'],'doc_id':pages[j]['id'],'rank':rank,'score':f'{score:.8g}'})
            if i%20==0 or i==len(queries):print('Ranked',i,'/',len(queries),flush=True)
    manifest={
        'experimental_status':'DIAGNOSTIC ONLY; NOT HUMAN-VALIDATED' if a.diagnostic else 'UNJUDGED RETRIEVAL RUNS; NOT PERFORMANCE RESULTS',
        'pages_sha256':digest(a.pages),'queries_sha256':digest(a.queries),
        'source_page_count':len(pages),'num_queries':len(queries),'num_chunks':len(chunks),
        'chunk_policy':{'tokenizer':TOKEN_PATTERN,'tokens':a.chunk_tokens,'overlap':a.chunk_overlap,'short_chunk_minimum':max(20,a.chunk_tokens//4)},
        'systems':{'BM25_PAGE':'k1=1.2 b=0.75 k3 implicit binary query; IDF log1p((N-df+.5)/(df+.5))',
                   'BM25_PASSAGE_MAX':'same BM25 on 180/45 token chunks, page=max chunk score',
                   'BM25_METADATA_DATE':'same raw BM25, but if provenance query has explicit ISO requested_issue_date, matching newspaper+date pages rank ahead; otherwise identical to BM25_PAGE',
                   'TFIDF_PAGE':'scikit-learn TfidfVectorizer sublinear_tf=True norm=l2',
                   'RRF_PAGE_PASSAGE_60':'RRF k=60 over BM25_PAGE and BM25_PASSAGE_MAX top100'},
        'zero_score_policy':'Always return top-k unique pages. Lexical zero-score ties broken in stable corpus order; do NOT discard unmatched questions.',
        'full_corpus_search':True,'all_systems_rank_against_identical_3608_page_corpus':len(pages)==3608,
        'max_rank':a.topk,'elapsed_sec':round(time.time()-t,3),'human_judgments':False,
        'model_tuning_using_phase24':False,'retrieval_semantic_dense_baseline_completed':False,
        'note':'As the queries have not been adjudicated, these are candidate rankings only.'}
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('DONE',round(time.time()-t,1),'seconds',out)

if __name__=='__main__':main()
