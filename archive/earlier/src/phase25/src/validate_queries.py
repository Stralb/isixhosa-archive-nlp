#!/usr/bin/env python3
"""Check collected query data; never treats self-attestation as independently verified."""
from __future__ import annotations
import argparse,csv,json,hashlib,collections,re,datetime
from pathlib import Path
REQUIRED={'query_id','query_text','source_scope','intent','language','author_id','topic_id','independent_author_attestation'}

def load(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser();p.add_argument('--queries',required=True);p.add_argument('--out',required=True);p.add_argument('--allow-incomplete',action='store_true');a=p.parse_args()
    items=load(a.queries)
    if not items:raise SystemExit('EMPTY QUERY CSV')
    if REQUIRED-set(items[0]):raise SystemExit(f'Missing columns: {sorted(REQUIRED-set(items[0]))}')
    issues=[];ids=set();unique=set()
    for i,r in enumerate(items,2):
        q=r['query_text'].strip(); key=r['query_id'].strip()
        if not key or key in ids:issues.append(f'line {i}: duplicate/blank query_id')
        ids.add(key)
        if not q:issues.append(f'line {i}: blank query_text')
        elif q.casefold() in unique:issues.append(f'line {i}: repeated query text')
        unique.add(q.casefold())
        if r['intent'] not in ('evidence','provenance','verbatim_reuse'):issues.append(f'line {i}: invalid intent')
        if r['source_scope'] not in ('IMVO','ISIG','BOTH','ANY'):issues.append(f'line {i}: invalid source')
        if not r['author_id']:issues.append(f'line {i}: missing author_id')
        if r.get('query_collection_mode','open_research_question') not in ('open_research_question','known_item'):issues.append(f'line {i}: invalid query_collection_mode')
        if r.get('known_reference_page_id') and not r.get('reference_basis'):issues.append(f'line {i}: known reference missing reference_basis')
        if r['intent']=='provenance':
            date=r.get('requested_issue_date','')
            try:
                if not re.fullmatch(r'\d{4}-\d{2}-\d{2}',date):raise ValueError('bad format')
                datetime.date.fromisoformat(date)
            except ValueError:issues.append(f'line {i}: provenance question requires real requested_issue_date YYYY-MM-DD')
        if r['independent_author_attestation']!='YES':issues.append(f'line {i}: human independence attestation absent (YES required)')
    digest=hashlib.sha256(Path(a.queries).read_bytes()).hexdigest()
    result={'input':str(Path(a.queries).resolve()),'sha256':digest,'n_rows':len(items),
            'n_valid_by_format':len(items) if not issues else None,'self_attested_only':True,
            'no_independent_authorship_verification_claimed':True,'errors':issues,
            'counts_by_source_intent':{f'{k[0]}:{k[1]}':v for k,v in collections.Counter((r['source_scope'],r['intent']) for r in items).items()}}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(result,indent=2))
    print(json.dumps({'queries':len(items),'errors':len(issues),'sha256':digest}))
    if issues and not a.allow_incomplete:raise SystemExit('VALIDATION FAILED: '+('; '.join(issues[:6])))

if __name__=='__main__': main()
