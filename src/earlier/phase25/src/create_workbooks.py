#!/usr/bin/env python3
"""Produce empty study worksheets. No human data or labels are manufactured."""
from __future__ import annotations
import csv, argparse, json
from pathlib import Path

QUERY_FIELDS=['query_id','query_text','source_scope','intent','language','author_id','topic_id','created_utc','independent_author_attestation','external_source_if_any','query_collection_mode','known_reference_page_id','reference_basis','requested_issue_date','notes']
JUDGE_FIELDS=['query_id','candidate_id','doc_id','relevance','provenance_match','verbatim_match','confidence','evidence_excerpt','assessor_notes']

def write_csv(p,rows,fields):
    with Path(p).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def make_rows(prefix,n_by_source_intent,author_count):
    out=[];i=0
    for source,intent,n in n_by_source_intent:
        for _ in range(n):
            i+=1
            out.append(dict(query_id=f'{prefix}{i:03d}',query_text='',source_scope=source,intent=intent,
                language='en',author_id=f'AUTHOR{(i-1)%author_count+1:02d}',topic_id=f'TOPIC{i:03d}',
                created_utc='',independent_author_attestation='',external_source_if_any='',query_collection_mode='open_research_question',known_reference_page_id='',reference_basis='',requested_issue_date='',notes=''))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    # Pilot isolated from final and cannot be promoted to confirmation.
    pilot=make_rows('PILOT', [('IMVO','evidence',7),('IMVO','provenance',3),('ISIG','evidence',7),('ISIG','provenance',3)],4)
    final=make_rows('Q',[('IMVO','evidence',25),('IMVO','provenance',15),('ISIG','evidence',25),('ISIG','provenance',15)],4)
    write_csv(out/'query_author_pilot_20.csv',pilot,QUERY_FIELDS)
    write_csv(out/'query_author_main_80.csv',final,QUERY_FIELDS)
    write_csv(out/'adjudicated_judgments_EMPTY.csv',[],['query_id','doc_id','relevance','provenance_match','verbatim_match','confidence','adjudication_notes'])
    (out/'TEMPLATE_STATUS.json').write_text(json.dumps({'n_pilot_slots':20,'n_main_slots':80,'queries_collected':0,
         'human_assessors_recruited':0,'relevance_judgments_collected':0,'status':'UNFILLED_PLANNING_TEMPLATES'},indent=2))
    print('Created empty pilot and main author worksheets: 20 + 80 planned slots, 0 actual questions.')

if __name__=='__main__':main()
