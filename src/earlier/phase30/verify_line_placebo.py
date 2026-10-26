#!/usr/bin/env python3
"""Independent data and result audit for post-result line-preserving placebo."""
from pathlib import Path
from collections import Counter
import csv,json
import sentencepiece as spm
D=Path(__file__).resolve().parent;P=D/'inputs'
r=json.loads((D/'LINE_PLACEBO_RESULTS.json').read_text())
original=list(csv.DictReader((D/'source_page_shared_decomposition.csv').open()))
new=list(csv.DictReader((D/'line_placebo_page_scores.csv').open()))
assert len(new)==len(original)==621
byoriginal={(t['file'],int(t['seed'])):t for t in original}
windows={w['file']:w for w in (json.loads(z) for z in (P/'windows.jsonl').read_text().splitlines()) if w['year']==1888}
for seed in range(3):
 original_lines=[]
 for source in ('IMVO','ISIG'):
  original_lines.extend((P/f'train_{source}_seed{seed}.txt').read_text().splitlines())
 placebo_lines=[]
 for c in ('A','B'):
  path=D/f'lineplacebo_{c}_seed{seed}.txt'; lines=path.read_text().splitlines();placebo_lines.extend(lines)
  assert sum(sum(ch.isalpha() for ch in line) for line in lines)==60000
 assert Counter(placebo_lines)==Counter(original_lines), (seed,'line integrity issue')
models={(source,k):spm.SentencePieceProcessor(model_file=str(D/f'LINE_PLACEBO_{source}_seed{k}.model')) for source in ('A','B') for k in range(3)}
assert all(p.vocab_size()==1000 for p in models.values())
for t in new:
 w=windows[t['file']];seed=int(t['seed']);o=byoriginal[t['file'],seed]
 assert w['issue']==t['issue'] and w['source']==t['source']
 assert sum(ch.isalpha() for ch in w['text'])==int(t['chars'])==int(o['all_alphabetic_chars'])
 pa=sum(len(models['A',seed].encode(tok)) for tok in w['tokens']);pb=sum(len(models['B',seed].encode(tok)) for tok in w['tokens'])
 assert pa==int(t['A_pieces']) and pb==int(t['B_pieces'])
 assert int(t['oriented_saved_pieces'])==int((pb-pa)*(1 if w['source']=='IMVO' else -1))
verification={'passed':True,'original_line_multiset_preserved_exactly':True,'placebo_model_counts':len(models),'test_rows_verified':len(new),'source_training_budget_each':60000,'subword_scores_exact':True,'placebo_sensitivity_gate_result':r['passes_sensitivity_gate']}
(D/'LINE_PLACEBO_VERIFICATION.json').write_text(json.dumps(verification,indent=2))
print(json.dumps(verification,indent=2))
