"""Prospectively documented 12-word-overlap exclusion sensitivity; no model changes."""
import json,collections,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from evaluate_historical_frozen import OCR,cleaner,forms,paragraphs,training_shingles,seen,manifest,models,alphabet,alpha_len
root=Path(__file__).resolve().parent
output={}
for book in ('ityala_lamawele','zemkinkomo_magwalandini'):
 for variant in ('raw','joined_line_hyphens'):
  counts=collections.Counter(); matched=0; total=0
  for line in (OCR/f'{book}_pages.jsonl').read_text().splitlines():
   rec=json.loads(line)
   for block in paragraphs(cleaner(rec['text'],variant)):
    ww=forms(block);covered=set();total+=len(ww)
    for idx in range(len(ww)-11):
     if tuple(ww[idx:idx+12]) in training_shingles:covered.update(range(idx,idx+12))
    matched+=len(covered)
    counts.update(w for i,w in enumerate(ww) if i not in covered)
  eligible={w:all((not c.isalpha() or c in alphabet) for c in w) for w in counts}
  for w in list(counts):
   if not eligible[w]:continue
   for model in models.values():
    if model.unk_id() in model.encode(w,out_type=int):
     eligible[w]=False;break
  s_all={w:ct for w,ct in counts.items() if eligible[w]}
  s_nov={w:ct for w,ct in s_all.items() if w not in seen}
  d_all=sum(alpha_len(w)*n for w,n in s_all.items());d_nov=sum(alpha_len(w)*n for w,n in s_nov.items())
  score={}
  for mix in manifest['mixes']:
   aval=[]; nval=[]
   for seed in manifest['seeds']:
    mod=models[(mix,seed)]
    aval.append(100*sum(n*len(mod.encode(w,out_type=int)) for w,n in s_all.items())/d_all)
    nval.append(100*sum(n*len(mod.encode(w,out_type=int)) for w,n in s_nov.items())/d_nov)
   score[str(mix)]={'all_by_seed':aval,'unseen_by_seed':nval,'all_mean':statistics.mean(aval),'unseen_mean':statistics.mean(nval)}
  pair=[v-u for v,u in zip(score['0.5']['unseen_by_seed'],score['1']['unseen_by_seed'])]
  output[f'{book}::{variant}']={'matched_excluded_words':matched,'total_original_words':total,'removed_fraction':matched/total,'remaining_words':sum(counts.values()),'scores':score,'parity_minus_opportunity_unseen_by_seed':pair,'parity_minus_opportunity_unseen_mean':statistics.mean(pair),'positive_seeds':sum(x>0 for x in pair)}
  print(book,variant,'removed',matched,'delta',statistics.mean(pair),'positive seeds',sum(x>0 for x in pair),flush=True)
(root/'EXCLUSION_SENSITIVITY.json').write_text(json.dumps(output,indent=2,ensure_ascii=False))
