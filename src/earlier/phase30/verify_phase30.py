#!/usr/bin/env python3
"""Independent integrity checks for Phase 30 saved corpus, models and score matrices."""
import collections, csv, hashlib, json, math
from pathlib import Path
import numpy as np
import sentencepiece as spm
from scipy.stats import spearmanr
D=Path(__file__).resolve().parent;P=D/'inputs'
r=json.load(open(D/'RESULTS.json'))
for line in (D/'FROZEN_SHA256.txt').read_text().splitlines():
 digest,name=line.split(maxsplit=1);x=P/name
 assert x.is_file() and hashlib.sha256(x.read_bytes()).hexdigest()==digest,(name,'input hash mismatch')
win=[json.loads(t) for t in (P/'windows.jsonl').read_text().splitlines()];w88={t['file']:t for t in win if t['year']==1888}
assert len(w88)==207
rows=list(csv.DictReader((D/'shared_word_frequency_vs_pieces.csv').open()))
models={(s,k):spm.SentencePieceProcessor(model_file=str(P/f'BPE_{s}_seed{k}.model')) for s in ('IMVO','ISIG') for k in range(3)}
assert all(m.vocab_size()==1000 for m in models.values())
max_pieces=0;max_freq=0
for k in range(3):
 train={s:collections.Counter((P/f'train_{s}_seed{k}.txt').read_text().split()) for s in ('IMVO','ISIG')}
 q=[t for t in rows if int(t['seed'])==k];xs=[];ys=[]
 for row in q:
  s=row['word'];a,b=train['IMVO'][s],train['ISIG'][s]
  fa=float(math.log((a+.5)/(b+.5)));ya=len(models['ISIG',k].encode(s))-len(models['IMVO',k].encode(s))
  max_pieces=max(max_pieces,abs(ya-int(row['isig_minus_imvo_pieces'])))
  max_freq=max(max_freq,abs(fa-float(row['train_frequency_log_ratio'])))
  assert a+b>=2
  xs.append(fa);ys.append(ya)
 corr=float(spearmanr(xs,ys).statistic)
 assert abs(corr-r['H30A'][k]['all']['rho'])<1e-12,(k,corr,r['H30A'][k]['all']['rho'])
assert max_pieces==0 and max_freq<1e-12

fixed=list(csv.DictReader((D/'source_page_shared_decomposition.csv').open()))
original=list(csv.DictReader((P/'tokenizer_page_scores.csv').open()))
old={(t['file'],int(t['seed'])):t for t in original}
assert len(fixed)==621
for row in fixed:
 key=row['file'],int(row['seed']);src=w88[row['file']]['source'];k=key[1]
 assert row['issue']==w88[row['file']]['issue'] and row['source']==src
 chars=sum(sum(c.isalpha() for c in t) for t in w88[row['file']]['tokens'])
 d=int(old[key]['ISIG_pieces'])-int(old[key]['IMVO_pieces'])
 if src=='ISIG':d=-d
 assert int(row['all_specialist_saved_pieces'])==d
 assert chars==int(row['all_alphabetic_chars'])

# Validate placebo train has required budget and both null model evaluations independently
placebomodels={(v,k):spm.SentencePieceProcessor(model_file=str(D/f'PLACEBO_{v}_seed{k}.model')) for v in ('A','B') for k in range(3)}
assert all(m.vocab_size()==1000 for m in placebomodels.values())
for k in range(3):
 cnt_pool=collections.Counter()
 for s in ('IMVO','ISIG'):cnt_pool.update((P/f'train_{s}_seed{k}.txt').read_text().split())
 cnt_control=collections.Counter()
 for v in ('A','B'):
  data=(D/f'placebo_{v}_seed{k}.txt').read_text()
  tokens=data.split();alpha=sum(sum(c.isalpha() for c in word) for word in tokens)
  assert alpha==59950, (v,k,alpha)
  assert all(len(line.split())<=90 for line in data.splitlines())
  cnt_control.update(tokens)
 assert all(n<=cnt_pool[w] for w,n in cnt_control.items())
placebo=list(csv.DictReader((D/'placebo_page_scores.csv').open()))
assert len(placebo)==621
for t in placebo:
 w=w88[t['file']];k=int(t['seed'])
 assert t['issue']==w['issue'] and t['source']==w['source']
 for v in ('A','B'):
  z=sum(len(placebomodels[v,k].encode(word)) for word in w['tokens'])
  assert z==int(t[f'placebo_{v}_pieces']), (t['file'],k,v)

# Recompute source-wise weighted observable from recorded scored pages.
check=[]
for k in range(3):
 tot=0;place=0
 for src in ('IMVO','ISIG'):
  q=[p for p in fixed if p['source']==src and int(p['seed'])==k]
  s=100*sum(int(t['all_specialist_saved_pieces']) for t in q)/sum(int(t['all_alphabetic_chars']) for t in q)
  qp=[p for p in placebo if p['source']==src and int(p['seed'])==k]
  p=100*sum(int(t['oriented_specialist_savings']) for t in qp)/sum(int(t['alphabetic_chars']) for t in qp)
  tot+=s;place+=p
 check.append((tot,place))
 assert abs(tot-r['H30C']['per_seed'][k]['actual_S'])<1e-12
 assert abs(place-r['H30C']['per_seed'][k]['random_source_S'])<1e-12

ver={'passed':True,'frozen_source_hashes_checked':len(list(P.glob('*'))),'frequency_rows':len(rows),'fixed_score_rows':len(fixed),'placebo_score_rows':len(placebo), 'recomputed_frequency_rho_by_seed':[float(spearmanr([float(z['train_frequency_log_ratio']) for z in rows if int(z['seed'])==k],[int(z['isig_minus_imvo_pieces']) for z in rows if int(z['seed'])==k]).statistic) for k in range(3)],'verified_seed_S_pairs':check,'model_counts':{'experimental':len(models),'placebo':len(placebomodels)},'max_piece_difference':max_pieces,'max_frequency_log_difference':max_freq}
(D/'VERIFICATION.json').write_text(json.dumps(ver,indent=2))
print(json.dumps(ver,indent=2))
