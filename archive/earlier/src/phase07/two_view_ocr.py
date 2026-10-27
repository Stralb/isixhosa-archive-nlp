#!/usr/bin/env python3
"""Tesseract PSM3 vs PSM6 paired diagnostic on frozen evenly spaced book pages."""
import collections, difflib, hashlib, io, json, os, re, subprocess, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import fitz
from PIL import Image
import sentencepiece as spm
ROOT=Path(__file__).resolve().parent
H=Path('/mnt/data/isiXhosa_historical_eval')
E=H/'isiXhosa_external_validation'
sys.path.insert(0,str(E))
from common import training_pages, forms, alpha_len

BOOK_PDF={'ityala_lamawele':'/mnt/data/whole.pdf','zemkinkomo_magwalandini':"/mnt/data/Zemnk'Inkomo.pdf"}
BOOKS=['ityala_lamawele','zemkinkomo_magwalandini']


def target_rows():
    tasks=[]
    for book in BOOKS:
        rows=[json.loads(x) for x in (H/'ocr'/f'{book}_pages.jsonl').read_text().splitlines()]
        n=len(rows)
        inds=[round((i+.5)*n/24-.5) for i in range(24)]
        assert len(set(inds))==24
        for ix in inds:
            r=rows[ix];tasks.append({'book':book,'pdf_page':r['pdf_page'],'side':r['side'],
                                      'original_text':r['text'],'sample_index':ix})
    return tasks

def worker(task):
    os.environ['OMP_THREAD_LIMIT']='1'
    book=task['book'];pageno=task['pdf_page'];side=task['side']
    with fitz.open(BOOK_PDF[book]) as pdf:
        pix=pdf[pageno-1].get_pixmap(matrix=fitz.Matrix(2.2,2.2),colorspace=fitz.csGRAY,alpha=False)
    im=Image.frombytes('L',(pix.width,pix.height),pix.samples)
    if side=='left':im=im.crop((0,0,im.width//2,im.height))
    elif side=='right':im=im.crop((im.width//2,0,im.width,im.height))
    buf=io.BytesIO();im.save(buf,format='PNG')
    proc=subprocess.run(['tesseract','stdin','stdout','-l','eng','--psm','6'],
                        input=buf.getvalue(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=100)
    if proc.returncode!=0:raise RuntimeError((book,pageno,proc.stderr[:300]))
    return dict(task,alternate_psm6=proc.stdout.decode('utf8','replace').strip())

def get_costfunc():
    manifest=json.loads((E/'models/training_manifest.json').read_text())
    assert manifest['mixes']==[0,0.5,1] and manifest['seeds']==[11,23,37,47,59]
    arr=[]
    for mix in manifest['mixes']:
        for seed in manifest['seeds']:
            rr=next(r for r in manifest['runs'] if r['seed']==seed and r['imvo_share']==mix)
            path=E/'models'/(rr['name']+'.model')
            assert hashlib.sha256(path.read_bytes()).hexdigest()==rr['model_sha256']
            arr.append(spm.SentencePieceProcessor(model_file=str(path)))
    alphabet=set(manifest['alphabet'])
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def f(w):
        if any(c.isalpha() and c not in alphabet for c in w):return None
        values=[]
        for p in arr:
            ids=p.encode(w,out_type=int)
            if p.unk_id() in ids:return None
            values.append(len(ids))
        return values
    return f

def filter_watermarks(text):
    # Same limited watermark deletion as frozen book evaluation.
    return re.sub(r'(?mi)^\s*Together in Excellence\s*$', '',
                  re.sub(r'(?mi)^\s*University of Fort Hare\s*$', '',text))

def score():
    path=ROOT/'two_view_psm6_pages.jsonl'
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    assert len(rows)==48
    seen={w for p in training_pages() for w in forms(p['text'])}
    cost=get_costfunc()
    out={'source':'same-PDF two-configuration Tesseract agreement proxy; no ground truth',
         'protocol_sha256':hashlib.sha256((ROOT/'TWO_VIEW_PRECOMMIT.md').read_bytes()).hexdigest(),
         'book_summary':{},'per_page':[]}
    for book in BOOKS:
        records=[r for r in rows if r['book']==book]
        buckets={kind:collections.Counter() for kind in ('all','agree','disagree')}
        checked=0
        for row in records:
            a=forms(filter_watermarks(row['original_text']))
            b=forms(filter_watermarks(row['alternate_psm6']))
            matches=set()
            matcher=difflib.SequenceMatcher(None,a,b,autojunk=False)
            for op,a0,a1,b0,b1 in matcher.get_opcodes():
                if op=='equal':matches.update(range(a0,a1))
            for i,w in enumerate(a):
                buckets['all'][w]+=1
                buckets['agree' if i in matches else 'disagree'][w]+=1
            out['per_page'].append({'book':book,'pdf_page':row['pdf_page'],'side':row['side'],
                                    'psm3_words':len(a),'psm6_words':len(b),'matched_words':len(matches),
                                    'unseen_psm3_matched':sum(w not in seen for i,w in enumerate(a) if i in matches),
                                    'unseen_psm3_unmatched':sum(w not in seen for i,w in enumerate(a) if i not in matches)})
            checked+=1
        summary={}
        for kind,cnt in buckets.items():
            n=sum(cnt.values());novel=sum(v for w,v in cnt.items() if w not in seen)
            all_num=[0]*15;nov_num=[0]*15; all_den=0;nov_den=0;sup_n=0
            for w,v in cnt.items():
                vals=cost(w)
                if vals is None:continue
                chars=alpha_len(w)*v;sup_n+=v;all_den+=chars
                for i,x in enumerate(vals):all_num[i]+=x*v
                if w not in seen:
                    nov_den+=chars
                    for i,x in enumerate(vals):nov_num[i]+=x*v
            if not (all_den and nov_den):continue
            by_mix={}; contrasts=[]
            for mix_ix,mix in enumerate((0,0.5,1)):
                by_mix[str(mix)]={'E_all_by_seed':[100*all_num[mix_ix*5+i]/all_den for i in range(5)],
                                  'E_unseen_by_seed':[100*nov_num[mix_ix*5+i]/nov_den for i in range(5)]}
            pp=by_mix['0.5']['E_unseen_by_seed'];oo=by_mix['1']['E_unseen_by_seed']
            summary[kind]={'word_occurrences':n,'unseen_occurrences':novel,'unseen_fraction':novel/n,
                           'supported_fraction':sup_n/n,'by_mix':by_mix,
                           'parity_minus_opportunity_unseen_by_seed':[a-b for a,b in zip(pp,oo)],
                           'parity_minus_opportunity_unseen_mean':sum(a-b for a,b in zip(pp,oo))/5}
        out['book_summary'][book]=summary
    (ROOT/'two_view_results.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
    print('SAVED',ROOT/'two_view_results.json')
    for book,summary in out['book_summary'].items():
        for kind,r in summary.items():print(book,kind,'n',r['word_occurrences'],'unseen%',round(100*r['unseen_fraction'],2),'50-100',round(r['parity_minus_opportunity_unseen_mean'],4))

def main():
    tasks=target_rows();assert len(tasks)==48
    existing={}
    path=ROOT/'two_view_psm6_pages.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            r=json.loads(line);existing[(r['book'],r['pdf_page'],r['side'])]=r
    remaining=[t for t in tasks if (t['book'],t['pdf_page'],t['side']) not in existing]
    print('task count',len(tasks),'remaining',len(remaining),flush=True)
    st=time.time()
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures={executor.submit(worker,task):task for task in remaining}
        with path.open('a',encoding='utf8') as save:
            for i,fut in enumerate(as_completed(futures),1):
                r=fut.result();save.write(json.dumps(r,ensure_ascii=False)+'\n');save.flush()
                if i%6==0:print('done',i,'/',len(remaining),'seconds',round(time.time()-st,1),flush=True)
    score()

if __name__=='__main__':main()
