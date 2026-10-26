#!/usr/bin/env python3
"""Independent result-file audits; does not rely on scientific narrative."""
from pathlib import Path
import collections,csv,hashlib,json,math,random
import numpy as np
import sentencepiece as spm
from sklearn.metrics import balanced_accuracy_score
from textblob import TextBlob
from textblob.en import sentiment
D=Path(__file__).parent
read=lambda name:[json.loads(l) for l in open(D/name,encoding='utf8')]
W=read('windows.jsonl');P=read('bilingual_pairs.jsonl'); assert len(W)==581 and len(P)==262
assert all(len(w['tokens'])==90 and w['text']==' '.join(w['tokens']) for w in W)
assert len({w['text'] for w in W})==len(W)
assert not ({w['issue'] for w in W if w['year'] in (1885,1887)} & {w['issue'] for w in W if w['year']==1888})
for fn in ['windows.jsonl','bilingual_pairs.jsonl']:
    sha=hashlib.sha256((D/fn).read_bytes()).hexdigest()
    assert sha==(D/(fn+'.sha256')).read_text().split()[0]
with open(D/'newspaper_test_predictions.csv') as f: clf=list(csv.DictReader(f))
for model in ('char_wb_3_5','word_unigram'):
    sub=[r for r in clf if r['model']==model]
    y=np.array([int(r['gold_imvo']) for r in sub]);p=np.array([int(r['pred_imvo']) for r in sub]);res=json.load(open(D/'newspaper_classifier_summary.json'))
    ans=next(r for r in res if r['model']==model)
    assert abs(balanced_accuracy_score(y,p)-ans['balanced_accuracy_page'])<1e-12
with open(D/'tokenizer_page_scores.csv') as f: score=list(csv.DictReader(f))
models={}
for src in ('IMVO','ISIG'):
    for seed in range(3):
        models[(src,seed)]=spm.SentencePieceProcessor(model_file=str(D/f'BPE_{src}_seed{seed}.model'))
for r in random.Random(26024).sample(score,30):
    page=next(w for w in W if w['file']==r['file'])
    assert int(r['alphabetic_chars'])==sum(c.isalpha() for c in page['text'])
    for src in ('IMVO','ISIG'):
        assert len(models[(src,int(r['seed']))].encode(page['text']))==int(r[f'{src}_pieces'])
with open(D/'sentiment_page_diagnostics.csv') as f: sr=list(csv.DictReader(f))
byfile={p['file']:p for p in P}
for r in random.Random(262624).sample(sr,30):
    pp=byfile[r['file']]
    for lg in ('en','xh'):
        val=sum(x in sentiment for x in pp[lg].split())/80
        assert abs(val-float(r[lg+'_coverage']))<1e-12
        assert abs(TextBlob(pp[lg]).sentiment.polarity-float(r[lg+'_polarity']))<1e-12
with open(D/'tokenizer_training_audit.csv') as f:
    for r in csv.DictReader(f):
        fp=D/f"train_{r['source']}_seed{r['seed']}.txt"
        assert hashlib.sha256(fp.read_bytes()).hexdigest()==r['sha256']
        assert sum(c.isalpha() for c in fp.read_text())==60000
p=json.load(open(D/'phase26_abc_results.json'))
assert len(p['temporal'])==4 and all(r['permutation_p']>0 for r in p['temporal'])
print('VERIFIED: 581 distinct 90-word windows; 262 bilingual page pairs; all BPE budgets/hashes; 30 random BPE score replays; 30 sentiment polarity/coverage replays; both classifier metrics; issue-disjoint 1888 transfer.')
