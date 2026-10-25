#!/usr/bin/env python3
"""Source-only Phase 1 design feasibility audit; never reads held-out text."""
import argparse, csv, hashlib, importlib.util, json, re, zipfile
from collections import Counter, defaultdict
from pathlib import Path

PAT = re.compile(r'^(IMVO|ISIG)_(\d{4})-(\d{2})-(\d{2})_p(\d{3})\.txt$')
TOK = re.compile(r'[a-z]+')
K=13
WINDOW=32

def sha(b):return hashlib.sha256(b).hexdigest()

def parse(zippath):
    pages={}; attempts=Counter(); foreign_dates=set()
    with zipfile.ZipFile(zippath) as z:
        for name in z.namelist():
            if name.startswith('__MACOSX/') or not name.endswith('.txt'):continue
            parent = name.split('/')[-2] if '/' in name else ''
            if parent not in ('EnTxt','XhTxt'):continue
            m=PAT.match(name.split('/')[-1])
            if not m:continue
            paper, year, mon, day, pg=m.groups(); year=int(year)
            # Metadata counts only for test; never read post-1887 content.
            if year>=1888:
                foreign_dates.add((paper,year,mon,day)); attempts['test_filenames_inspected']+=1;continue
            if paper!='IMVO' or year not in (1885,1887):continue
            key=m.group(0); rec=pages.setdefault(key,dict(file=key,year=year,month=int(mon),date=f'{year}-{mon}-{day}', issue=f'{paper}_{year}-{mon}-{day}',views={}))
            arr=TOK.findall(z.read(name).decode('utf-8-sig',errors='replace').lower())
            rec['views']['en' if parent=='EnTxt' else 'xh']=arr
            attempts['train_or_val_view_files_read']+=1
        attempts['heldout_view_content_reads']=0
    assert all(set(r['views'])=={'en','xh'} for r in pages.values())
    train=sorted((r for r in pages.values() if r['year']==1885 or (r['year']==1887 and r['month']<=9)),key=lambda r:r['file'])
    val=sorted((r for r in pages.values() if r['year']==1887 and r['month']>=10),key=lambda r:r['file'])
    assert train and val
    assert not (set(x['issue'] for x in train) & set(x['issue'] for x in val))
    return train,val,foreign_dates,dict(attempts)

def enumerate_units(pages):
    units=[]; perpage=[]
    for rec in pages:
        arr=[]
        for v in ('xh','en'):
            w=rec['views'][v]
            n=len(w) if len(w)>=16 else 0
            arr.append(n)
            if n:
                units.append({'file':rec['file'],'issue':rec['issue'],'view':v,'words':w,'nwords':n})
        perpage.append(arr)
    assert all(sum(i)>0 for i in perpage)
    return units,perpage

def shingling(train):
    # Exact 13-token shingle repeated across DIFFERENT publication issues.
    owners=defaultdict(set)
    units,pp=enumerate_units(train)
    for u in units:
        words=u['words']
        for i in range(len(words)-K+1):
            owners[tuple(words[i:i+K])].add(u['issue'])
    duplicated={g for g,issues in owners.items() if len(issues)>1}
    out=[]
    for u in units:
        w=u['words']; n=len(w)
        bad=[tuple(w[i:i+K]) in duplicated for i in range(n-K+1)]
        starts=max(n-WINDOW+1,1)
        keep=0
        for i in range(starts):
            end=min(i+WINDOW,n)
            if not any(bad[i:max(i,end-K+1)]):keep+=1
        out.append({'filename':u['file'],'view':u['view'],'total_candidate_windows':starts,'duplicate_free_windows':keep,'words':n,'issue':u['issue']})
    return out, len(duplicated),len(owners)

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--phase33',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    train,val,testdates,reads=parse(args.archive)
    units,pp=enumerate_units(train)
    val_units,val_pp=enumerate_units(val)
    s_en=sum(n[1] for n in pp);s_xh=sum(n[0] for n in pp)
    p_page=sum(n[1]/sum(n) for n in pp)/len(pp)
    p_word=s_en/(s_en+s_xh)
    # PP view/page joint is exactly original Phase33 P: (1/N)*(n_v/n_total).
    # WW view/page joint is exactly W: n_v/sum_{page,v}n_v.
    # PW is Phase33 M. WP is the missing factorial arm.
    fact=[]
    for alloc,p_en in [('page_like',p_page),('word_like',p_word)]:
        for within in ('page_conditional','word_conditional'):
            arr=[]
            for view_i,view in [(0,'xh'),(1,'en')]:
                qv=p_en if view=='en' else 1-p_en
                prior=[(n[view_i]/sum(n))/len(pp) if within=='page_conditional' else n[view_i] for n in pp]
                total=sum(prior)
                assert total>0
                for ix,wt in enumerate(prior):
                    if wt>0:arr.append((ix,view,qv*wt/total))
            totalprob=sum(r[2] for r in arr)
            p_en_check=sum(r[2] for r in arr if r[1]=='en')
            assert abs(totalprob-1)<1e-10 and abs(p_en_check-p_en)<1e-10
            arm={('page_like','page_conditional'):'PP',('page_like','word_conditional'):'PW',('word_like','page_conditional'):'WP',('word_like','word_conditional'):'WW'}[(alloc,within)]
            fact.append({'arm':arm,'language_allocation':alloc,'page_conditional':within,'view_draw_english_probability':p_en,'nonzero_page_view_cells':len(arr), 'probability_sum':totalprob})
    with (out/'factorial_probabilities.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=fact[0]);wr.writeheader();wr.writerows(fact)
    # Predict source-only eligibility of repeated-content exclusion. Do not use this to optimize a result.
    du,ngramdu,ngramtotal=shingling(train)
    # Validation leakage screen: these strings are allowable for development,
    # but a verbatim repeated 13-word run must not masquerade as generalization.
    tr_by_view={v:set() for v in ('en','xh')}
    for u in units:
        w=u['words']
        tr_by_view[u['view']].update(tuple(w[i:i+K]) for i in range(len(w)-K+1))
    val_contamination={}
    for view in ('en','xh'):
        rows=[u for u in val_units if u['view']==view]
        repeated=total=clean=any_affected=0
        for u in rows:
            w=u['words'];hit=[tuple(w[i:i+K]) in tr_by_view[view] for i in range(len(w)-K+1)]
            total+=len(hit);repeated+=sum(hit);any_affected+=any(hit)
            for i in range(max(len(w)-WINDOW+1,1)):
                end=min(i+WINDOW,len(w))
                if not any(hit[i:max(i,end-K+1)]):clean+=1
        val_contamination[view]={'validation_page_views':len(rows),'validation_13grams':total,
           '13grams_repeated_from_training':repeated,'fraction_repeated':repeated/total,
           'page_views_containing_repeat':any_affected,'uncontaminated_windows_32word':clean}
    with (out/'TRAIN_DUPLICATION_ELIGIBILITY.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=du[0]);wr.writeheader();wr.writerows(du)
    byview={}
    for view in ('xh','en'):
        rows=[x for x in du if x['view']==view]
        byview[view]={
          'eligible_page_views':len(rows),
          'page_views_with_a_duplicate_free_32word_window':sum(x['duplicate_free_windows']>0 for x in rows),
          'candidate_window_count':sum(x['total_candidate_windows'] for x in rows),
          'duplicate_free_window_count':sum(x['duplicate_free_windows'] for x in rows),
          'duplicate_free_window_fraction':sum(x['duplicate_free_windows'] for x in rows)/sum(x['total_candidate_windows'] for x in rows),
        }
    # Prior-phase frozen checksums + summary refer to prior independently verified outcomes, not new test access.
    prior=Path(args.phase33)
    main_reg=prior/'PRECOMMIT.md';ext_reg=prior/'EXTENSION_PRECOMMIT.md'
    with (prior/'INDEPENDENT_VERIFICATION.json').open() as f:prev=json.load(f)
    with (prior/'EXTENDED_VERIFICATION.json').open() as f:prev2=json.load(f)
    prior_audit_path=prior/'outputs'/'training_audit.csv'
    assert prev['pass'] and prev2['pass']
    with prior_audit_path.open() as f:prev_audit=list(csv.DictReader(f))
    assert len(prev_audit)==15
    reg_hash=sha(main_reg.read_bytes())
    assert reg_hash==prev['sampling_precommit_sha256']
    result={
      'audit_scope':'TRAIN and VALIDATION text only; 1888+ filenames only',
      'archive_sha256':sha(Path(args.archive).read_bytes()),
      'source_zip_member_content_access':reads,
      'training':{'years':'1885 and 1887 Jan-Sep','pages':len(train),'issues':len(set(r['issue'] for r in train)),
         'eligible_page_views':len(units),'alphabetic_word_tokens_en':s_en,'alphabetic_word_tokens_xh':s_xh,
         'page_like_expected_english_draw_share':p_page,'word_like_expected_english_draw_share':p_word,
         'view_draw_delta_word_minus_page':p_word-p_page,
         'train_page_filenames_sha256':sha(('\n'.join(r['file'] for r in train)+'\n').encode())},
      'validation':{'years':'1887 Oct-Dec','pages':len(val),'issues':len(set(r['issue'] for r in val)),
         'eligible_page_views':len(val_units),'validation_page_filenames_sha256':sha(('\n'.join(r['file'] for r in val)+'\n').encode()),
         'word_tokens_en':sum(n[1] for n in val_pp),'word_tokens_xh':sum(n[0] for n in val_pp)},
      'test_inventory_only':{'year_range':'1888-1894','candidate_issue_filenames':len(testdates)},
      'factorial':fact,
      'duplicate_train_only':{'repeat_shingles_13word_across_distinct_issues':ngramdu,'total_shingle_types':ngramtotal,
         'eligible_by_view':byview},
      'train_to_validation_verbatim_13gram_overlap':val_contamination,
      'prior_phase33':{'sha256_precommit':reg_hash,'sha256_extension_precommit':sha(ext_reg.read_bytes()),
          'original_primary_models':len(prev_audit),'original_page_score_rows':prev['page_rows'],
          'independent_verification_pass':prev['pass'],'extended_verification_pass':prev2['pass']},
      'split_checks':{'train_val_page_overlap':bool(set(r['file'] for r in train)&set(r['file'] for r in val)),
          'train_val_issue_overlap':bool(set(r['issue'] for r in train)&set(r['issue'] for r in val)),
          'heldout_view_text_was_read':False},
      'phase1_not_a_new_performance_test':True
    }
    (out/'feasibility_audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ('training','validation','duplicate_train_only','split_checks')},indent=2))

if __name__=='__main__':main()
