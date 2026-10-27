#!/usr/bin/env python3
"""Score two *real* independent human annotation exports; refuse missing labels.

Do NOT treat pilot-candidate distribution as corpus prevalence or detector recall.
No generated or automatic labels are substituted for human judgements.
"""
import argparse, collections, json
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ROOT=Path(__file__).resolve().parent
RELATION={'verbatim_reuse','edited_or_OCR_variant_reuse','shared_formula_only',
          'same_topic_without_reuse','unrelated','uncertain'}
CONTENT={'commercial_ad','administrative_masthead','official_notice',
         'editorial_report_letter','religious_literary','other','uncertain'}

def read_records(path):
    result={}
    with open(path,encoding='utf8') as f:
        for lno,line in enumerate(f,1):
            if not line.strip():continue
            item=json.loads(line);name=item.get('annotation_id')
            if not name or name in result:raise ValueError(f'{path}:{lno} missing or repeated ID')
            result[name]=item
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('reviewer_a')
    ap.add_argument('reviewer_b')
    ap.add_argument('--output',default=str(ROOT/'human_agreement.json'))
    ar=ap.parse_args()
    gold_ids={x['annotation_id'] for x in read_records(ROOT/'annotation_pairs_blind.jsonl').values()}
    a=read_records(ar.reviewer_a);b=read_records(ar.reviewer_b)
    if set(a)!=gold_ids or set(b)!=gold_ids:
        raise ValueError('Reviewer files must contain exactly the blinded queue IDs; missing/extra cases are not imputed.')
    for path,d in [(ar.reviewer_a,a),(ar.reviewer_b,b)]:
        incomplete=[]
        for case,r in d.items():
            if r.get('relationship_label') not in RELATION or r.get('content_label') not in CONTENT or \
                r.get('confidence') not in {'high','medium','low'} or r.get('image_checked') not in {'yes','no'}:
                incomplete.append(case)
        if incomplete:
            raise ValueError(f'Incomplete or invalid human labels in {path}: {len(incomplete)} of {len(d)} records, examples {incomplete[:3]}. No scores calculated.')
    ids=sorted(gold_ids)
    summary={'n_pairs':len(ids),'human_label_status':'two_complete_review_files_present',
             'quality_warning':'If full-page scans were not checked, content labels remain provisional.',
             'prevalence_inference':'not allowed - deliberately stratified, non-probability retrieval sample',
             'retrieval_recall_inference':'not allowed - no probability sample of missed reuse',
             'results':{}}
    for field in ('relationship_label','content_label'):
        x=[a[i][field] for i in ids];y=[b[i][field] for i in ids]
        labels=sorted(set(x)|set(y))
        summary['results'][field]={
            'raw_agreement':sum(xx==yy for xx,yy in zip(x,y))/len(ids),
            'cohen_kappa':float(cohen_kappa_score(x,y,labels=labels)),
            'classes':labels,'confusion_matrix':confusion_matrix(x,y,labels=labels).tolist(),
            'annotator_a_distribution':dict(collections.Counter(x)),
            'annotator_b_distribution':dict(collections.Counter(y))}
    for label,arr in [('annotator_a',a),('annotator_b',b)]:
        summary[label+'_scanned_page_fraction']=sum(arr[i]['image_checked']=='yes' for i in ids)/len(ids)
    Path(ar.output).write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
