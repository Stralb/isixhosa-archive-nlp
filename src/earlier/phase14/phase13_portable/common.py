"""Shared source reading and deterministic surface-form operations."""
import io, os, re, zipfile, unicodedata
from pathlib import Path

ORIGINAL_ZIP = '/mnt/data/TxtLangFiles1870-1894_TB.zip'
PAGE_RE = re.compile(r'(IMVO|ISIG)_(\d{4})-(\d{2})-(\d{2})_p\d+\.txt$')
SURFACE_RE = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", re.UNICODE)

def forms(text):
    return SURFACE_RE.findall(text.casefold())

def alpha_len(text):
    return sum(c.isalpha() for c in text)

def paragraphs(text):
    return re.split(r'\n\s*\n+', text)

def iter_pages(zip_path=ORIGINAL_ZIP, years=('1887',), months=None):
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(z.namelist()):
            if '/XhTxt/' not in name or not name.endswith('.txt') or name.startswith('__MACOSX'): continue
            m = PAGE_RE.search(name)
            if not m: continue
            pub, year, month, day=m.groups()
            if years and year not in years: continue
            if months and month not in months: continue
            yield {'source':pub,'year':year,'month':month,'day':day,'id':Path(name).name,
                   'text':z.read(name).decode('utf-8',errors='replace')}

def training_pages(zip_path=ORIGINAL_ZIP):
    return list(iter_pages(zip_path,months={f'{i:02d}' for i in range(1,10)}))

def dev_pages(zip_path=ORIGINAL_ZIP):
    return list(iter_pages(zip_path,months={'10','11','12'}))

def paragraph_shingles(text,k=12):
    for par in paragraphs(text):
        tokens = forms(par)
        for i in range(len(tokens)-k+1): yield tuple(tokens[i:i+k])
