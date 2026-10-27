#!/usr/bin/env python3
"""Optional pretrained multilingual E5 dense retrieval. NOT runnable without model weights.

Weights must be obtained lawfully and explicitly supplied via --model-path. This script
never calls an online endpoint or fabricates scores. Model version/checksum recorded.
"""
from __future__ import annotations
import argparse,csv,json,hashlib,time
from pathlib import Path
import numpy as np
try:
 from retrieve import read_pages,read_queries,chunks_from_pages,digest,index_order
except ImportError:
 from src.retrieve import read_pages,read_queries,chunks_from_pages,digest,index_order

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pages',required=True);ap.add_argument('--queries',required=True)
    ap.add_argument('--model-path',required=True,help='Existing local snapshot directory of intfloat/multilingual-e5-small')
    ap.add_argument('--out',required=True);ap.add_argument('--topk',type=int,default=20)
    ap.add_argument('--chunk-tokens',type=int,default=180);ap.add_argument('--chunk-overlap',type=int,default=45)
    ap.add_argument('--batch-size',type=int,default=24);ap.add_argument('--diagnostic',action='store_true')
    a=ap.parse_args()
    modelpath=Path(a.model_path)
    if not modelpath.is_dir():raise SystemExit('MODEL NOT FOUND: supply downloaded, versioned multilingual-e5-small weights')
    try:from sentence_transformers import SentenceTransformer
    except ImportError:raise SystemExit('Install optional sentence-transformers dependency; no dense scores created')
    pages=read_pages(a.pages);queries=read_queries(a.queries,diagnostic=a.diagnostic)
    chunks,owners,_=chunks_from_pages(pages,a.chunk_tokens,a.chunk_overlap)
    model=SentenceTransformer(str(modelpath),device='cpu',local_files_only=True)
    # E5 requires prefixes for asymmetric retrieval, regardless of query language.
    docemb=model.encode(['passage: '+c for c in chunks],batch_size=a.batch_size,
          show_progress_bar=True,normalize_embeddings=True,convert_to_numpy=True)
    qemb=model.encode(['query: '+q['query_text'] for q in queries],batch_size=a.batch_size,
          show_progress_bar=True,normalize_embeddings=True,convert_to_numpy=True)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    with (out/'dense_runs.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,['system','query_id','doc_id','rank','score']);w.writeheader()
        for q,vec in zip(queries,qemb):
            scores=np.asarray(docemb@vec).reshape(-1)
            agg=np.full(len(pages),-np.inf,dtype=np.float32)
            np.maximum.at(agg,owners,scores)
            for rank,j in enumerate(index_order(agg,a.topk,positive_only=False),1):
                w.writerow({'system':'DENSE_MULTILINGUAL_E5_SMALL','query_id':q['query_id'],
                            'doc_id':pages[j]['id'],'rank':rank,'score':f'{float(agg[j]):.8g}'})
    sha=[]
    for p in sorted(modelpath.rglob('*')):
        if p.is_file():sha.append({'file':str(p.relative_to(modelpath)),'sha256':digest(p)})
    (out/'dense_manifest.json').write_text(json.dumps({'status':'EXECUTED', 'model':'intfloat/multilingual-e5-small',
       'model_files_sha256':sha,'pages_sha256':digest(a.pages),'queries_sha256':digest(a.queries),
       'chunk_tokens':a.chunk_tokens,'overlap':a.chunk_overlap,'prefix_query':'query: ',
       'prefix_passage':'passage: ','pooling':'normalized sentence-transformer embedding; max chunk cosine per page',
       'n_docs':len(pages),'n_chunks':len(chunks),'n_queries':len(queries),'no_finetuning':True,
       'warning':'Historical isiXhosa competence unverified; no claim of language coverage'},indent=2))

if __name__=='__main__':main()
