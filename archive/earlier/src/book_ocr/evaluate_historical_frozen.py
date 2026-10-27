"""Frozen external historical evaluation. Never trains, reselects, or modifies tokenizers."""
import sys, os, json, re, hashlib, collections, statistics, math
from pathlib import Path
import sentencepiece as spm

ROOT=Path(__file__).resolve().parent; PKG=ROOT/'isiXhosa_external_validation'
sys.path.insert(0,str(PKG))
from common import training_pages,forms,alpha_len,paragraphs
from score_external import BOOKS

OCR=ROOT/'ocr'; MODELS=PKG/'models'
manifest=json.loads((MODELS/'training_manifest.json').read_text())
selection=json.loads((MODELS/'selection_frozen.json').read_text())
assert selection['raw_parity_selected']==0.5 and selection['opportunity_selected']==1
assert set(manifest['seeds'])=={11,23,37,47,59} and manifest['mixes']==[0,0.5,1]
assert BOOKS=={'ityala_lamawele','zemkinkomo_magwalandini'}

models={}
for rr in manifest['runs']:
    model_path=MODELS/(rr['name']+'.model')
    assert hashlib.sha256(model_path.read_bytes()).hexdigest()==rr['model_sha256'],('Frozen artifact mismatch',rr['name'])
    models[(rr['imvo_share'],rr['seed'])]=spm.SentencePieceProcessor(model_file=str(model_path))

newspapers=training_pages(os.environ.get('NEWSPAPER_ZIP','/mnt/data/TxtLangFiles1870-1894_TB.zip'))
seen=set(w for p in newspapers for w in forms(p['text']))
training_shingles=set()
for p in newspapers:
    for par in paragraphs(p['text']):
        words=forms(par)
        training_shingles.update(tuple(words[i:i+12]) for i in range(len(words)-11))
print('Original training pages',len(newspapers),'distinct forms',len(seen),'12-grams',len(training_shingles),flush=True)
alphabet=set(manifest['alphabet'])


def cleaner(raw,variant):
    # An exact scanner watermark string is not part of the historical original.
    text=re.sub(r'(?mi)^\s*University of Fort Hare\s*$', '', raw)
    text=re.sub(r'(?mi)^\s*Together in Excellence\s*$', '', text)
    if variant=='joined_line_hyphens':
        text=re.sub(r'(?<=[a-zA-Z])-\s*\n\s*(?=[a-zA-Z])','',text)
    return text


def boot_ci(deltas,seed=20261023):
    # Descriptive bootstrap over FIVE tokenizer seeds; do not infer cross-book generalisability.
    import random
    rng=random.Random(seed)
    arr=[]
    for _ in range(20000):
        arr.append(sum(deltas[rng.randrange(len(deltas))] for _ in deltas)/len(deltas))
    arr.sort()
    return [arr[int(.025*len(arr))],arr[int(.975*len(arr))]]


def evaluate(book,variant):
    pages=[json.loads(line) for line in (OCR/f'{book}_pages.jsonl').read_text().splitlines()]
    counts=collections.Counter(); pagescores=[]
    matchwords=0; totalwords=0
    ocr_alphabet_chars=0
    watermark_removed=0
    perpage_counts=[]
    for page in pages:
        txt=cleaner(page['text'],variant)
        watermark_removed+=len(page['text'])-len(txt)
        arr=forms(txt)
        cnt=collections.Counter(arr)
        perpage_counts.append(cnt)
        counts.update(cnt)
        ocr_alphabet_chars+=alpha_len(txt)
        for block in paragraphs(txt):
            arrblock=forms(block);totalwords+=len(arrblock)
            covered=set()
            for i in range(len(arrblock)-11):
                if tuple(arrblock[i:i+12]) in training_shingles: covered.update(range(i,i+12))
            matchwords+=len(covered)
    supported={}
    modelcost={}
    for word in counts:
        eligible=bool(word) and all(not c.isalpha() or c in alphabet for c in word)
        if eligible:
            encs=[]
            for mix in manifest['mixes']:
                for seed in manifest['seeds']:
                    model=models[(mix,seed)]
                    ids=model.encode(word,out_type=int)
                    if model.unk_id() in ids:eligible=False
                    encs.append(len(ids))
            if eligible:modelcost[word]=encs
        supported[word]=eligible
    count_all=sum(counts.values())
    novel_occ=sum(n for w,n in counts.items() if w not in seen)
    supported_all=[(w,n) for w,n in counts.items() if supported[w]]
    supported_novel=[(w,n) for w,n in counts.items() if supported[w] and w not in seen]
    denom_all=sum(len(w.replace('-','').replace("'",'').replace('’',''))*n for w,n in supported_all)
    # exact: character counting includes only alpha, not apostrophe, hyphen etc
    denom_all=sum(alpha_len(w)*n for w,n in supported_all)
    denom_novel=sum(alpha_len(w)*n for w,n in supported_novel)
    denom_total=sum(alpha_len(w)*n for w,n in counts.items())
    row={'book':book,'variant':variant,'pages_and_sides':len(pages), 'all_word_occurrences':count_all,
         'all_alpha_chars':denom_total,'raw_alpha_chars':ocr_alphabet_chars,
         'distinct_forms':len(counts),'unseen_occurrences':novel_occ,'unseen_occurrence_fraction':novel_occ/count_all,
         'supported_occurrences':sum(n for _,n in supported_all),'supported_char_fraction':denom_all/denom_total,
         'supported_novel_occurrences':sum(n for _,n in supported_novel),'supported_novel_alpha_chars':denom_novel,
         'detected_12word_overlap_fraction':matchwords/totalwords,'detected_12word_overlap_words':matchwords,'removed_watermark_characters':watermark_removed,
         'by_mix':{},'seed_disagreement':{},'pagewise':[]}
    costs={}
    for mix_ix,mix in enumerate(manifest['mixes']):
        aa=[];uu=[]
        for seed_ix,seed in enumerate(manifest['seeds']):
            ix=5*mix_ix+seed_ix
            totalcost=sum(modelcost[w][ix]*n for w,n in supported_all)
            novcost=sum(modelcost[w][ix]*n for w,n in supported_novel)
            aa.append(100*totalcost/denom_all)
            uu.append(100*novcost/denom_novel)
        row['by_mix'][str(mix)]={'all_E_by_seed':aa,'all_E_mean':statistics.mean(aa),'unseen_E_by_seed':uu,'unseen_E_mean':statistics.mean(uu)}
        costs[mix]=(aa,uu)
    pair=[p-o for p,o in zip(costs[.5][1],costs[1][1])]
    pair_all=[p-o for p,o in zip(costs[.5][0],costs[1][0])]
    row['primary_parity_minus_opportunity']={'unseen_by_seed':pair,'unseen_mean':statistics.mean(pair),
         'unseen_positive_seeds':sum(x>0 for x in pair),'descriptive_seed_bootstrap_ci95':boot_ci(pair),
         'all_by_seed':pair_all,'all_mean':statistics.mean(pair_all)}
    for p,pcnt in zip(pages,perpage_counts):
        if not pcnt: continue
        supported_nov=[(w,n) for w,n in pcnt.items() if supported[w] and w not in seen]
        supnovchars=sum(n*alpha_len(w) for w,n in supported_nov)
        per_mix={}
        if supnovchars:
            for mix_ix,mix in enumerate(manifest['mixes']):
                per_mix[str(mix)]=[100*sum(modelcost[w][mix_ix*5+i]*n for w,n in supported_nov)/supnovchars for i in range(5)]
        row['pagewise'].append({'pdf_page':p['pdf_page'],'side':p['side'],'supported_unseen_alpha_chars':supnovchars,'unseen_E_by_mix_seed':per_mix})
    return row


def main():
    res={'source':'User-supplied historical scanned PDFs','frozen_manifest_sha256':hashlib.sha256((MODELS/'training_manifest.json').read_bytes()).hexdigest(),
         'frozen_selection_sha256':hashlib.sha256((MODELS/'selection_frozen.json').read_bytes()).hexdigest(),
         'model_count':len(models), 'training_pages':len(newspapers),'training_distinct_forms':len(seen),
         'training_paragraph_12_grams':len(training_shingles),
         'models_retrained':False,
         'scores':[]}
    for book in ['ityala_lamawele','zemkinkomo_magwalandini']:
        for variant in ['raw','joined_line_hyphens']:
            print('Scoring',book,variant,flush=True)
            res['scores'].append(evaluate(book,variant))
            (ROOT/'HISTORICAL_BOOK_RESULTS.json').write_text(json.dumps(res,indent=2,ensure_ascii=False))
            r=res['scores'][-1]
            print('mean costs 50/100',r['by_mix']['0.5']['unseen_E_mean'],r['by_mix']['1']['unseen_E_mean'], 'contrast',r['primary_parity_minus_opportunity']['unseen_mean'],'positive seeds',r['primary_parity_minus_opportunity']['unseen_positive_seeds'],flush=True)
    print('DONE',flush=True)
if __name__=='__main__':main()
