"""Exploratory exact-overlap traceback after frozen external scoring; OCR misses near matches."""
import json,sys,os,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from evaluate_historical_frozen import OCR,training_pages,forms,paragraphs,cleaner
train=collections.defaultdict(set)
for p in training_pages(os.environ.get('NEWSPAPER_ZIP','/mnt/data/TxtLangFiles1870-1894_TB.zip')):
    for paragraph in paragraphs(p['text']):
        ww=forms(paragraph)
        for i in range(len(ww)-11): train[tuple(ww[i:i+12])].add(p['id'])
books={}
for book in ['ityala_lamawele','zemkinkomo_magwalandini']:
    hits=[];word_sum=0;page_total=0
    for line in (OCR/f'{book}_pages.jsonl').read_text().splitlines():
        p=json.loads(line);n=0
        for block in paragraphs(cleaner(p['text'],'raw')):
            arr=forms(block);word_sum+=len(arr)
            for i in range(len(arr)-11):
                ng=tuple(arr[i:i+12]);which=train.get(ng)
                if which:
                    n+=1
                    if len(hits)<1000:
                        hits.append({'pdf_page':p['pdf_page'],'side':p['side'],'shingle':' '.join(ng), 'newspaper_pages':sorted(which)[:4]})
        if n:page_total+=1
    books[book]={'number_exact_12gram_windows':sum(1 for x in hits), 'pages_with_matches':page_total,
                 'examples':hits[:15], 'pages_counter':dict(collections.Counter(h['pdf_page'] for h in hits).most_common(15))}
out=Path(__file__).resolve().parent/'EXACT_OVERLAP_TRACEBACK.json';out.write_text(json.dumps(books,indent=2,ensure_ascii=False))
for b,vals in books.items():
 print(b,'windows',vals['number_exact_12gram_windows'],'pages',vals['pages_with_matches'],'examples',vals['examples'][:3])
