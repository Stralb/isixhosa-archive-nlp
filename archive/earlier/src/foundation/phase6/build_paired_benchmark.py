#!/usr/bin/env python3
"""Build unlabelled issue-paired historical isiXhosa reuse annotation instrument.

Only 1887 Jan-Sep data is accessed. The prospective scientific rules are in
PHASE6_PRECOMMIT.md. Text categories are retrieval strata, NOT human ground truth.
"""
import collections
import hashlib
import html
import itertools
import json
import random
import re
import zipfile
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent
DATA = Path('/mnt/data/TxtLangFiles1870-1894_TB.zip')
RAW = Path('/mnt/data/1. TxtFiles1870-1894_TB.zip')
RNG = random.Random(20261022)
N = 12
TARGET = 10
RX_FN = re.compile(r'^(IMVO|ISIG)_1887-(\d\d)-(\d\d)_p(\d+)\.txt$')
RX_WORD = re.compile(r'[^\W\d_]+', re.UNICODE)
STRATA = ('exact_persistent', 'exact_transient', 'near_no_exact', 'random_no_exact')


def digest(tokens):
    return hashlib.blake2b(' '.join(tokens).encode('utf8'), digest_size=12).hexdigest()


def load_paragraphs():
    paragraphs, source_pages = [], {}
    with zipfile.ZipFile(DATA) as z:
        for path in sorted(z.namelist()):
            if '/XhTxt/' not in path or '/._' in path or not path.endswith('.txt'):
                continue
            fn = path.rsplit('/', 1)[-1]
            m = RX_FN.fullmatch(fn)
            if not m or int(m.group(2)) > 9:
                continue
            src, month, day, page_slot = m.groups()
            text = z.read(path).decode('utf8', 'replace')
            source_pages[fn] = text
            for pos, block in enumerate(re.split(r'\n\s*\n', text)):
                words = RX_WORD.findall(block.casefold())
                if len(words) < 12:
                    continue
                paragraphs.append({
                    'source': src,
                    'month': '1887-' + month,
                    'issue': f'{src}_1887-{month}-{day}',
                    'filename': fn, 'slot': int(page_slot),
                    'paragraph_index': pos, 'raw': block.strip(),
                    'tokens': words,
                    'normalized': ' '.join(words),
                })
    return paragraphs, source_pages


def build_shingles(paragraphs):
    index = collections.defaultdict(set)
    perparagraph = []
    for i, p in enumerate(paragraphs):
        hashes = set(digest(p['tokens'][j:j + N]) for j in range(len(p['tokens']) - N + 1))
        perparagraph.append(hashes)
        for h in hashes:
            index[h].add(i)
    meta = {}
    for h, ids in index.items():
        issues = {paragraphs[i]['issue'] for i in ids}
        if len(issues) < 2:
            continue
        months = {paragraphs[i]['month'] for i in ids}
        slot_issues = collections.defaultdict(set)
        for i in ids:
            slot_issues[paragraphs[i]['slot']].add(paragraphs[i]['issue'])
        persistent = len(issues) >= 3 and len(months) >= 2 and max(len(x) for x in slot_issues.values()) >= 3
        meta[h] = 'exact_persistent' if persistent else 'exact_transient'
    return index, perparagraph, meta


def candidate_pairs(paragraphs, index, meta):
    pairs = collections.defaultdict(lambda: collections.Counter())
    for h, cls in meta.items():
        grouped = collections.defaultdict(set)
        for i in index[h]:
            p = paragraphs[i]
            grouped[p['source']].add(i)
        for src, ids in grouped.items():
            # Each pair is counted once per matching shingle, not per occurrence.
            for a, b in itertools.combinations(sorted(ids), 2):
                if paragraphs[a]['issue'] == paragraphs[b]['issue']:
                    continue
                pairs[(src, a, b)][cls] += 1
    result = collections.defaultdict(list)
    for (src, a, b), count in pairs.items():
        cls = 'exact_persistent' if count['exact_persistent'] else 'exact_transient'
        pa, pb = paragraphs[a], paragraphs[b]
        shared = count['exact_persistent'] + count['exact_transient']
        jaccard = shared / max(1, len(pa['tokens']) + len(pb['tokens']) - 2 * (N - 1) - shared)
        score = np.log1p(shared) + (shared / max(1, min(len(pa['tokens']), len(pb['tokens']) - N + 1)))
        result[(src, cls)].append({'a': a, 'b': b, 'shared_shingles': shared,
                                   'similarity': float(jaccard), 'rank_score': float(score)})
    return result


def near_pairs(paragraphs, perparagraph, src, nqueries=700):
    pool = [i for i, p in enumerate(paragraphs)
            if p['source'] == src and len(p['tokens']) >= 18 and len(p['raw']) >= 65
            and not (perparagraph[i] & all_repeated)]
    # Source-specific candidate set, not external retrieval data. A fixed-size
    # exploratory retrieval sample avoids quadratic all-pairs matrices.
    if not pool:
        return [], 0
    rng = random.Random(20261022 + (1 if src == 'IMVO' else 2))
    queries = rng.sample(pool, min(nqueries, len(pool)))
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 4), max_features=25000,
                                 min_df=2, dtype=np.float32, sublinear_tf=True)
    raw_texts = [paragraphs[i]['normalized'] for i in pool]
    X = vectorizer.fit_transform(raw_texts)
    index_by_p = {v: j for j, v in enumerate(pool)}
    out = []
    for q in queries:
        v = X[index_by_p[q]] @ X.T
        if v.nnz == 0:
            continue
        # Use the top lexical neighbours that are from other publication dates.
        order = np.argsort(v.data)[-30:][::-1]
        for j in order:
            local_j = int(v.indices[j])
            score = float(v.data[j])
            b = pool[local_j]
            if paragraphs[b]['issue'] == paragraphs[q]['issue'] or b == q:
                continue
            if perparagraph[q] & perparagraph[b]:
                continue
            if score < 0.24 or score > 0.99:
                continue
            a, b = sorted((q, b))
            out.append({'a': a, 'b': b, 'shared_shingles': 0,
                        'similarity': score, 'rank_score': score})
            break
    seen = set()
    dedup = []
    for r in sorted(out, key=lambda r: (-r['rank_score'], r['a'], r['b'])):
        key = (r['a'], r['b'])
        if key not in seen:
            dedup.append(r)
            seen.add(key)
    return dedup, len(dedup)


def random_pairs(paragraphs, perparagraph, src, ntrials=1400):
    pool = [i for i, p in enumerate(paragraphs)
            if p['source'] == src and len(p['tokens']) >= 18 and len(p['raw']) >= 65
            and not (perparagraph[i] & all_repeated)]
    if len(pool) < 2:
        return []
    out = []
    for _ in range(ntrials):
        a, b = RNG.sample(pool, 2)
        pa, pb = paragraphs[a], paragraphs[b]
        if pa['issue'] == pb['issue'] or perparagraph[a] & perparagraph[b]:
            continue
        if max(len(pa['tokens']), len(pb['tokens'])) > 2.5 * min(len(pa['tokens']), len(pb['tokens'])):
            continue
        a, b = sorted((a, b))
        out.append({'a': a, 'b': b, 'shared_shingles': 0,
                    'similarity': None, 'rank_score': RNG.random()})
    return out


def sim_set(p):
    # Not a linguistic similarity score; only a conservative queue-diversity check.
    tokens = p['tokens']
    return set(tuple(tokens[j:j+3]) for j in range(len(tokens) - 2))


def select(source, cls, records, pars, n=TARGET):
    # Greedy, deterministic issue/page diversity selection. Never silently pad
    # a source with near-identical shingle fragments.
    rng = random.Random(20261022 + len(source) * 113 + len(cls) * 71)
    rows = records[:]
    rng.shuffle(rows)
    if cls.startswith('exact'):
        rows.sort(key=lambda r: (-r['rank_score'], r['a'], r['b']))
        # Interleave strong and moderate matches so the queue is not all titles.
        top, mid = rows[:max(n * 25, 200)], rows[max(n * 25, 200):]
        rng.shuffle(top)
        rows = top + mid
    elif cls == 'near_no_exact':
        rows.sort(key=lambda r: -r['rank_score'])
        window = rows[:max(150, n * 30)]
        rng.shuffle(window)
        rows = window + rows[max(150, n * 30):]
    else:
        rng.shuffle(rows)
    chosen = []
    issuetotal = collections.Counter()
    pagetotal = collections.Counter()
    used_anchor = set()
    used_pair = set()
    anchorsets = []
    # Hard caps per stratum. The queue is allowed to be short.
    for row in rows:
        a, b = row['a'], row['b']
        if (a, b) in used_pair or a in used_anchor or b in used_anchor:
            continue
        pa, pb = pars[a], pars[b]
        if pa['issue'] == pb['issue'] or pa['source'] != pb['source'] or pa['source'] != source:
            continue
        if issuetotal[pa['issue']] >= 2 or issuetotal[pb['issue']] >= 2:
            continue
        if pagetotal[pa['filename']] >= 1 or pagetotal[pb['filename']] >= 1:
            continue
        words = sim_set(pa)
        if any(len(words & prior)/max(1, len(words | prior)) > .72 for prior in anchorsets):
            continue
        chosen.append(row)
        used_anchor.update((a,b))
        used_pair.add((a,b))
        for p in (pa,pb):
            issuetotal[p['issue']] += 1
            pagetotal[p['filename']] += 1
        anchorsets.append(words)
        if len(chosen) == n:
            break
    return chosen


def data_row(src, typ, item, pars):
    pa, pb = pars[item['a']], pars[item['b']]
    identity = '|'.join((src, typ, pa['filename'], str(pa['paragraph_index']),
                         pb['filename'], str(pb['paragraph_index'])))
    id_part = hashlib.sha256(identity.encode('utf8')).hexdigest()[:10].upper()
    ann_id = 'PAIR_' + id_part
    def compact(p):
        return {'source': p['source'], 'issue': p['issue'], 'page_file': p['filename'],
                'paragraph_index': p['paragraph_index'], 'word_count': len(p['tokens']),
                'paragraph_original': p['raw'],
                'full_page_context_file': 'source_pages/' + p['filename']}
    blind = {'annotation_id': ann_id, 'passage_a': compact(pa), 'passage_b': compact(pb),
             'relationship_label': '', 'content_label': '', 'confidence': '',
             'image_checked': '', 'notes': ''}
    key = {'annotation_id': ann_id, 'retrieval_stratum': typ,
           'source': src, 'paragraph_a_global_index': item['a'],
           'paragraph_b_global_index': item['b'],
           'number_shared_inparagraph_12grams': item['shared_shingles'],
           'automated_similarity': item['similarity']}
    return blind, key


def export_html(rows):
    # Static local preview. Deliberately does not assign any labels automatically.
    css = '''body{font:15px system-ui,Arial;max-width:1160px;margin:30px auto;padding:0 20px;color:#222} h1{font-size:26px} .rec{border:1px solid #ccc;padding:22px;margin:28px 0;border-radius:8px;break-inside:avoid} .two{display:grid;grid-template-columns:1fr 1fr;gap:18px}.p{border:1px solid #eee;padding:16px;background:#fafafa;white-space:pre-wrap;overflow:auto;max-height:360px}.meta{font-size:12px;color:#666;margin-bottom:8px}.note{background:#f5f5f5;padding:10px;margin:20px 0}.form{display:flex;gap:15px;flex-wrap:wrap;margin:14px 0}label{font-weight:600;font-size:13px}select,input,textarea{font:inherit;padding:5px;max-width:100%} textarea{width:95%}@media(max-width:850px){.two{grid-template-columns:1fr}}@media print{.rec{page-break-inside:avoid}.p{max-height:none}}'''
    ops = {'relationship_label': ('', 'verbatim_reuse', 'edited_or_OCR_variant_reuse', 'shared_formula_only', 'same_topic_without_reuse', 'unrelated', 'uncertain'),
           'content_label': ('', 'commercial_ad', 'administrative_masthead', 'official_notice', 'editorial_report_letter', 'religious_literary', 'other', 'uncertain'),
           'confidence': ('', 'high', 'medium', 'low'),
           'image_checked': ('', 'yes', 'no')}
    body=[]
    for i,r in enumerate(rows):
        h=['<section class="rec" data-ann="'+r['annotation_id']+'"><h2>'+str(i+1)+'. '+r['annotation_id']+'</h2><div class="two">']
        for side in ('passage_a','passage_b'):
            p=r[side]
            h.append('<article><div class="meta">'+html.escape(side.upper())+' | '+html.escape(p['issue'])+' | '+html.escape(p['page_file'])+' | paragraph '+str(p['paragraph_index'])+'</div>')
            h.append('<div class="p">'+html.escape(p['paragraph_original'])+'</div>')
            h.append('<p class="meta">Full text context: <a target="_blank" href="source_pages/'+html.escape(p['page_file'])+'">XhTxt page</a> | <a target="_blank" href="original_full_pages/'+html.escape(p['page_file'])+'">original extracted page text</a></p></article>')
        h.append('</div><div class="form">')
        for fld,opts in ops.items():
            sel='<label>'+fld+': <select data-field="'+fld+'">'+''.join('<option value="'+html.escape(o)+'">'+html.escape(o or '-- choose --')+'</option>' for o in opts)+'</select></label>'
            h.append(sel)
        h.append('</div><label>Notes: <textarea data-field="notes" rows="2" placeholder="Cite evidence or uncertainty"></textarea></label></section>')
        body.append(''.join(h))
    js='''const sections=[...document.querySelectorAll('.rec')];const reviewer=new URLSearchParams(location.search).get('reviewer')||'UNSET';const storeKey='isixhosa_phase6_annotations_v1_'+reviewer;let saved={};try{saved=JSON.parse(localStorage.getItem(storeKey)||'{}');}catch(e){}; for(const s of sections){const obj=saved[s.dataset.ann]||{};for(const e of s.querySelectorAll('[data-field]')){e.value=obj[e.dataset.field]||'';e.addEventListener('change',persist);e.addEventListener('input',persist);}}function persist(){let result={};for(const s of sections){let v={};for(const e of s.querySelectorAll('[data-field]')){v[e.dataset.field]=e.value;}result[s.dataset.ann]=v;}try{localStorage.setItem(storeKey,JSON.stringify(result));}catch(e){}}function exportReview(){persist();let lines=sections.map(s=>{let data={};for(const e of s.querySelectorAll('[data-field]'))data[e.dataset.field]=e.value;return JSON.stringify({annotation_id:s.dataset.ann,...data});}).join('\\n')+'\\n';let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines],{type:'application/x-ndjson;charset=utf-8'}));a.download='annotator_labels_'+(new URLSearchParams(location.search).get('reviewer')||'UNSET')+'.jsonl';a.click();URL.revokeObjectURL(a.href);}'''
    intro='<h1>Historical isiXhosa: paired passage review (UNLABELLED)</h1><div class="note">Independent reviewers: inspect both paragraphs AND complete source pages. Automatic retrieval category is hidden. Do not infer genre from this snippet alone. No page scans are bundled; choose image_checked=no until separately verified. Your annotations are stored in this browser where supported: click <b>Export annotations</b> to save a copy. Do not share labels between reviewers until both are complete.</div><button onclick="exportReview()">Export annotations as JSONL</button>'
    html_path=ROOT/'annotation_review.html'
    html_path.write_text('<!DOCTYPE html><html lang="en"><meta charset="utf-8"><title>Unlabelled isiXhosa passage review</title><style>'+css+'</style><body>'+intro+''.join(body)+'<script>'+js+'</script></body></html>',encoding='utf8')


def main():
    pars, src_pages = load_paragraphs()
    idx, hashes_by_par, m = build_shingles(pars)
    global all_repeated
    all_repeated = set(m)
    pairs = candidate_pairs(pars, idx, m)
    selected = []
    poolstats={}
    for src in ('IMVO','ISIG'):
        for typ in STRATA:
            if typ.startswith('exact'):
                pool = pairs.get((src,typ), [])
                candidate_len = len(pool)
            elif typ == 'near_no_exact':
                pool, candidate_len = near_pairs(pars, hashes_by_par, src)
            else:
                pool = random_pairs(pars, hashes_by_par, src)
                candidate_len = len(pool)
            chosen = select(src,typ,pool,pars)
            selected.extend((src,typ,record) for record in chosen)
            poolstats[f'{src}_{typ}']={'candidate_pairs':candidate_len,'selected_pairs':len(chosen),
                     'unique_anchor_pages':len({pars[r['a']]['filename'] for r in chosen}),
                     'unique_issues_both_sides':len({pars[i]['issue'] for r in chosen for i in (r['a'],r['b'])}),
                     'source_pages_both_sides':len({pars[i]['filename'] for r in chosen for i in (r['a'],r['b'])}),
                     'max_examples_from_issue':max(collections.Counter(pars[i]['issue'] for r in chosen for i in (r['a'],r['b'])).values(),default=0)}
            print(src,typ,len(chosen),'from',candidate_len,flush=True)
    blind,key=[],[]
    for src,typ,item in selected:
        b,k=data_row(src,typ,item,pars);blind.append(b);key.append(k)
    paired=list(zip(blind,key))
    RNG.shuffle(paired)
    blind,key=map(list,zip(*paired)) if paired else ([],[])
    with (ROOT/'annotation_pairs_blind.jsonl').open('w',encoding='utf8') as f:
        for p in blind:f.write(json.dumps(p,ensure_ascii=False)+'\n')
    with (ROOT/'annotation_pair_heuristic_key_PRIVATE.jsonl').open('w',encoding='utf8') as f:
        for p in key:f.write(json.dumps(p,ensure_ascii=False)+'\n')
    support=ROOT/'source_pages'
    support.mkdir(exist_ok=True)
    referenced={p[side]['page_file'] for p in blind for side in ('passage_a','passage_b')}
    for fn in referenced:
        (support/fn).write_text(src_pages[fn],encoding='utf8')
    # Both original pages and language-extracted pages are available for context;
    # original full pages are not image scans and may include other languages.
    raw_dir=ROOT/'original_full_pages'
    raw_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(RAW) as z:
        for path in z.namelist():
            name=path.rsplit('/',1)[-1]
            if name in referenced and '/._' not in path and path.endswith('.txt'):
                (raw_dir/name).write_bytes(z.read(path))
    export_html(blind)
    qc=[]
    for b,k in zip(blind,key):
        a,c = pars[k['paragraph_a_global_index']], pars[k['paragraph_b_global_index']]
        assert a['source']==c['source']==k['source']
        assert a['issue'] != c['issue']
        assert a['raw']==b['passage_a']['paragraph_original']
        assert c['raw']==b['passage_b']['paragraph_original']
        shared=hashes_by_par[k['paragraph_a_global_index']] & hashes_by_par[k['paragraph_b_global_index']]
        if k['retrieval_stratum'].startswith('exact'):
            assert len(shared)>0
        else:assert len(shared)==0
        assert k['retrieval_stratum'] not in json.dumps(b)
        qc.append((k['source'],k['retrieval_stratum']))
    summary={'status':'UNLABELLED_REVIEW_INSTRUMENT_NOT_GROUND_TRUTH',
             'source_scope':'1887 January-September only, XhTxt, in-paragraph 12grams',
             'input_data_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),
             'precommit_sha256':hashlib.sha256((ROOT/'PHASE6_PRECOMMIT.md').read_bytes()).hexdigest(),
             'original_pages_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),
             'number_pairs':len(blind),'number_distinct_source_pages':len(referenced),
             'number_distinct_issues':len({b[side]['issue'] for b in blind for side in ('passage_a','passage_b')}),
             'strata':poolstats, 'qc_passed':len(qc),
             'no_expert_labels':True,'no_page_scans_in_supplied_zip':True,
             'no_corpus_recall_claim':True,'no_corpus_prevalence_claim':True}
    (ROOT/'annotation_queue_QA.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print('QA SUMMARY',json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    main()
