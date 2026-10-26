"""Deterministic OCR of user-supplied historical book PDFs; see HISTORICAL_EVAL_FROZEN_PROTOCOL.md."""
import os, json, hashlib, io, time, re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import fitz
from PIL import Image
import subprocess

DOCS={}
BOOK_FILES={'ityala_lamawele':os.environ.get('ITYALA_PDF','/mnt/data/whole.pdf'), 'zemkinkomo_magwalandini':os.environ.get('ZEM_PDF',"/mnt/data/Zemnk'Inkomo.pdf")}

def init():
    os.environ['OMP_THREAD_LIMIT']='1'
    for k,v in BOOK_FILES.items(): DOCS[k]=fitz.open(v)

def process(task):
    book,pageno,side=task
    doc=DOCS[book]
    p=doc[pageno]
    pix=p.get_pixmap(matrix=fitz.Matrix(2.2,2.2),colorspace=fitz.csGRAY,alpha=False)
    im=Image.frombytes('L',(pix.width,pix.height),pix.samples)
    if side=='left': im=im.crop((0,0,im.width//2,im.height))
    elif side=='right': im=im.crop((im.width//2,0,im.width,im.height))
    bio=io.BytesIO();im.save(bio,format='PNG')
    cmd=['tesseract','stdin','stdout','-l','eng','--psm','3']
    proc=subprocess.run(cmd,input=bio.getvalue(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=100)
    if proc.returncode!=0:raise RuntimeError(f'{book} p{pageno+1} {side} OCR failed {proc.stderr[:100]}')
    txt=proc.stdout.decode('utf-8','replace').strip()
    return {'book':book,'pdf_page':pageno+1,'side':side,'pixels':[im.width,im.height],
            'text':txt,'characters':len(txt),'wordlike':len(re.findall(r'[A-Za-z]{2,}',txt)),
            'ocr_sha256':hashlib.sha256(txt.encode()).hexdigest()}

def main():
    output=Path(__file__).resolve().parent/'ocr';output.mkdir(parents=True,exist_ok=True)
    tasks=[]
    doc=fitz.open(BOOK_FILES['ityala_lamawele'])
    for i in range(3,86):
        if i==3: tasks.append(('ityala_lamawele',i,'right'))
        elif doc[i].rect.width/doc[i].rect.height>0.90:
            tasks.extend([('ityala_lamawele',i,'left'),('ityala_lamawele',i,'right')])
        else: tasks.append(('ityala_lamawele',i,'whole'))
    doc.close()
    tasks.extend(('zemkinkomo_magwalandini',i,'whole') for i in range(10,520))
    manifest={'protocol':'HISTORICAL_EVAL_FROZEN_PROTOCOL.md','ocr_cmd':'tesseract stdin stdout -l eng --psm 3; 2.2x raster','input_pdfs':{},'n_tasks':len(tasks),'tasks':{}}
    for book,p in BOOK_FILES.items():
        h=hashlib.sha256();
        with open(p,'rb') as fh:
            for chunk in iter(lambda:fh.read(1<<20),b''):h.update(chunk)
        manifest['input_pdfs'][book]={'path':p,'sha256':h.hexdigest(),'bytes':Path(p).stat().st_size}
    start=time.monotonic(); results={}
    checkpoint=output/'ocr_partial_pages.jsonl'
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding='utf8').splitlines():
            if not line.strip(): continue
            rec=json.loads(line);results[(rec['book'],rec['pdf_page']-1,rec['side'])]=rec
    remaining=[t for t in tasks if t not in results]
    print('OCR loaded',len(results),'saved; remaining',len(remaining),flush=True)
    with ProcessPoolExecutor(max_workers=3,initializer=init) as pool:
        future_map={pool.submit(process,t):t for t in remaining}
        with checkpoint.open('a',encoding='utf8') as saved:
            for count,f in enumerate(as_completed(future_map),1):
                rec=f.result();task=(rec['book'],rec['pdf_page']-1,rec['side']); results[task]=rec
                saved.write(json.dumps(rec,ensure_ascii=False)+'\n'); saved.flush()
                if count%35==0:print('OCR finished',len(results),'/',len(tasks),'elapsed',round(time.monotonic()-start,1),'s',flush=True)
    for book in BOOK_FILES:
        rows=[results[t] for t in tasks if t[0]==book]
        for idx,row in enumerate(rows):row['position']=idx+1
        (output/f'{book}_pages.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
        combined='\n\n'.join(r['text'] for r in rows)
        (output/f'{book}_raw.txt').write_text(combined,encoding='utf8')
        manifest['tasks'][book]={'pages_and_sides':len(rows),'raw_alpha_chars':sum(c.isalpha() for c in combined),'raw_characters':len(combined),
                                 'short_pages_lt_150_chars':sum(len(r['text'])<150 for r in rows),'ocr_text_sha256':hashlib.sha256(combined.encode()).hexdigest()}
    manifest['elapsed_seconds']=round(time.monotonic()-start,1)
    (output/'OCR_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print('DONE',json.dumps(manifest['tasks'],indent=2),flush=True)

if __name__=='__main__':main()
