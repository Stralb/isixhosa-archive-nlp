#!/usr/bin/env python3
"""Intent-aware IR judging and scoring utilities (no external dependencies).

This is infrastructure, not a validated new retrieval metric or a new experiment.
The toolkit never infers semantic relevance or any human labels from text similarity.

Usage:
    python evaluation_toolkit.py prepare --queries my_queries.csv --runs system_runs.csv --out annotation
    python evaluation_toolkit.py score --queries my_queries.csv --runs system_runs.csv \
       --qrels adjudicated_judgments.csv --out evaluation_results.json
"""
import argparse,csv,hashlib,json,math,random,statistics
from collections import Counter,defaultdict
from pathlib import Path

QUERY_FIELDS=['query_id','query_text','source_scope','intent','language','author_id','topic_id']
RUN_FIELDS=['system','query_id','doc_id','rank']
QREL_FIELDS=['query_id','doc_id','relevance','provenance_match','verbatim_match','confidence','adjudication_notes']
INTENTS={'evidence','provenance','verbatim_reuse'}


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def require_fields(filename,arr,fields):
    if not arr:raise ValueError(f'{filename} has no rows')
    missing=set(fields)-set(arr[0]);
    if missing:raise ValueError(f'{filename} missing columns: {sorted(missing)}')


def write_csv(filename,headers,items):
    with Path(filename).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,headers,extrasaction='ignore');w.writeheader();w.writerows(items)


def read_inputs(queries,runs):
    q=rows(queries); r=rows(runs)
    require_fields(queries,q,QUERY_FIELDS);require_fields(runs,r,RUN_FIELDS)
    qby={}
    for row in q:
        key=row['query_id'].strip()
        if not key or key in qby:raise ValueError('empty or duplicate query_id')
        if row['intent'] not in INTENTS:raise ValueError(f'Invalid intent for {key}: {row["intent"]}')
        if not row['query_text'].strip():raise ValueError(f'Empty query for {key}')
        qby[key]=row
    bysystem=defaultdict(dict)
    for item in r:
        system=item['system'].strip();qid=item['query_id'].strip();doc=item['doc_id'].strip()
        if qid not in qby or not system or not doc:raise ValueError('Unknown qid or blank system/doc in runs')
        try: rank=int(item['rank'])
        except ValueError:raise ValueError(f'Invalid rank: {item}')
        if rank<1:raise ValueError('Rank must be >=1')
        if rank in bysystem[(system,qid)]:raise ValueError('Duplicate system/query/rank')
        bysystem[(system,qid)][rank]=doc
    for k,d in bysystem.items():
        rr=sorted(d)
        if rr!=list(range(1,len(rr)+1)):raise ValueError(f'Nonconsecutive ranks: {k}')
        if len(set(d.values()))!=len(d):raise ValueError(f'Duplicate doc in a run: {k}')
    return qby,bysystem


def make_pool(args):
    qby,runs=read_inputs(args.queries,args.runs)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    annot=[];secret=[]
    for qid,q in sorted(qby.items()):
        seen=defaultdict(dict)
        for (system,run_qid),rankmap in runs.items():
            if run_qid!=qid:continue
            for rank,doc in rankmap.items():
                if rank<=args.depth:seen[doc][system]=rank
        candidates=sorted(seen)
        salt=int.from_bytes(hashlib.sha256(f'{args.seed}|{qid}'.encode()).digest()[:8],'big')
        random.Random(salt).shuffle(candidates)
        for i,doc in enumerate(candidates,1):
            masked=f'{qid}_C{i:03d}'
            entry={'query_id':qid,'candidate_id':masked,'query_text':q['query_text'],
                'source_scope':q['source_scope'],'intent':q['intent'],'language':q['language'],
                'document_id':doc,'relevance':'','provenance_match':'','verbatim_match':'',
                'confidence':'','adjudication_notes':''}
            annot.append(entry)
            secret.append({'query_id':qid,'candidate_id':masked,'document_id':doc,
                           'included_by_system_rank_json':json.dumps(seen[doc],sort_keys=True)})
    if not annot: raise ValueError('No retrieved candidates; cannot prepare judgments')
    for assessor in [s.strip() for s in args.assessors.split(',') if s.strip()]:
        write_csv(out/f'judgments_{assessor}.csv',list(annot[0]),annot)
    write_csv(out/'sealed_pool_provenance.csv',list(secret[0]),secret)
    (out/'README.txt').write_text(
        f'{len(qby)} submitted queries (authorship NOT independently verified); {len(annot)} pooled documents. '
        'Each assessor must receive ONLY their judgments file, NOT sealed_pool_provenance.csv. '
        'Retrieve and inspect entire document/nearby context from the authoritative archive. '
        'Do not infer relevance from document IDs or system rankings. '
        'Judgment fields: relevance 0=not relevant, 1=partly answer-bearing, 2=clearly answer-bearing, '
        'U=uncertain; provenance_match 0/1/U; verbatim_match 0/1/U; confidence low/medium/high. '
        'Two independent judgment sheets need manual adjudication before they are combined into '
        'adjudicated_judgments.csv. Do NOT use this toolkit as a substitute for human annotation.\n')
    print(json.dumps({'queries':len(qby),'pooled_query_document_pairs':len(annot),'files':str(out)}))


def decode_label(value,valid,key):
    val=value.strip()
    if val in ('','U','?'):return None
    if val not in valid:raise ValueError(f'Invalid {key}: {value}')
    return int(val)


def score(args):
    qby,runs=read_inputs(args.queries,args.runs)
    qrels={};judge=rows(args.qrels);require_fields(args.qrels,judge,QREL_FIELDS)
    for rec in judge:
        key=(rec['query_id'].strip(),rec['doc_id'].strip())
        if key in qrels:raise ValueError(f'Duplicate final adjudicated qrel {key}')
        if key[0] not in qby:raise ValueError(f'Qrel query not known: {key}')
        qrels[key]={
          'semantic':decode_label(rec['relevance'],{'0','1','2'},'relevance'),
          'edition':decode_label(rec['provenance_match'],{'0','1'},'provenance'),
          'reuse':decode_label(rec['verbatim_match'],{'0','1'},'verbatim')}
    output=[];systems=sorted({s for s,q in runs})
    for system in systems:
        for qid,q in sorted(qby.items()):
            order=[d for rank,d in sorted(runs.get((system,qid),{}).items())]
            if not order:continue
            first=qrels.get((qid,order[0]),{})
            semantic=first.get('semantic');edition=first.get('edition');reuse=first.get('reuse')
            intent=q['intent']
            aligned={'evidence':None if semantic is None else int(semantic>=1),
                     'provenance':edition,'verbatim_reuse':reuse}[intent]
            full=order[:10]
            all_judged=bool(len(full)==10 and all((qid,doc) in qrels and qrels[(qid,doc)]['semantic'] is not None for doc in full))
            ndcg=None
            if all_judged:
                gains=[2**qrels[(qid,d)]['semantic']-1 for d in full]
                dcg=sum(g/math.log2(i+2) for i,g in enumerate(gains))
                # Pooled ideal: only judged pool, not exhaustive whole archive.
                ideal=sorted([2**x['semantic']-1 for (oq,doc),x in qrels.items() if oq==qid and x['semantic'] is not None],reverse=True)[:10]
                maxdcg=sum(g/math.log2(i+2) for i,g in enumerate(ideal))
                ndcg=dcg/maxdcg if maxdcg else None
            output.append({'system':system,'query_id':qid,'source_scope':q['source_scope'],
                          'intent':intent,'top1_doc_id':order[0],
                          'top1_judged_intent':aligned,'top1_judged_semantic':None if semantic is None else int(semantic>=1),
                          'top1_judged_provenance':edition,'top1_judged_verbatim':reuse,
                          'pooled_ndcg10_if_fully_judged':ndcg})
    summaries=[]
    for system in systems:
      for intent in sorted(INTENTS):
        part=[r for r in output if r['system']==system and r['intent']==intent]
        if not part:continue
        valid=[r['top1_judged_intent'] for r in part if r['top1_judged_intent'] is not None]
        ndcgs=[r['pooled_ndcg10_if_fully_judged'] for r in part if r['pooled_ndcg10_if_fully_judged'] is not None]
        summaries.append({'system':system,'intent':intent,'n_queries':len(part),
            'n_judged_top1':len(valid),'top1_judgment_coverage':len(valid)/len(part),
            'intent_aligned_top1_on_judged':sum(valid)/len(valid) if valid else None,
            'n_fully_judged_top10':len(ndcgs),
            'pooled_ndcg10_on_fully_judged':statistics.mean(ndcgs) if ndcgs else None})
    result={'type':'human_adjudication_evaluation','warnings':[
      'Do not treat unjudged retrieved documents as nonrelevant.',
      'Top-1 on judged outcomes is conditional on coverage; report coverage.',
      'Pooled nDCG is not an exhaustive relevance evaluation; label it explicitly.',
      'No human data were gathered or verified by this software.',
      'Source/intents must not be collapsed into one score without a documented target query distribution.'
      ],'summaries':summaries,'per_query':output}
    Path(args.out).write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps({'systems':systems,'summaries':summaries},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='cmd',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--queries',required=True);prep.add_argument('--runs',required=True)
    prep.add_argument('--out',required=True);prep.add_argument('--depth',type=int,default=10)
    prep.add_argument('--seed',type=int,default=20261024);prep.add_argument('--assessors',default='A,B');prep.set_defaults(func=make_pool)
    sc=sub.add_parser('score');sc.add_argument('--queries',required=True);sc.add_argument('--runs',required=True)
    sc.add_argument('--qrels',required=True);sc.add_argument('--out',required=True);sc.set_defaults(func=score)
    args=p.parse_args();args.func(args)

if __name__=='__main__':main()
