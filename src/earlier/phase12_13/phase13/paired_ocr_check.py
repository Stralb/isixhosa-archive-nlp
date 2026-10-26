"""Post hoc, different-OCR-setting stability check (NOT certificate validation)."""
import json,sys,collections
from pathlib import Path
import sentencepiece as spm
BASE=Path('/mnt/data/isiXhosa_phase13_ocr_certification')
sys.path.insert(0,'/mnt/data/isiXhosa_historical_eval/isiXhosa_external_validation')
from common import training_pages,forms
R=Path('/mnt/data/isiXhosa_ocr_counterfactual_phase7/two_view_psm6_pages.jsonl')
M=Path('/mnt/data/isiXhosa_historical_eval/isiXhosa_external_validation/models')
train={w for row in training_pages() for w in forms(row['text'])}
models={(mix,seed):spm.SentencePieceProcessor(model_file=str(M/f'bpe_imvo_{mix:03d}_seed_{seed}.model')) for mix in [0,50,100] for seed in [11,23,37,47,59]}
source=collections.defaultdict(lambda:{'original_text':[],'alternative_text':[]})
for line in R.read_text().splitlines():
 x=json.loads(line)
 old=x.get('original_text',x.get('text1',''))
 new=x.get('alternative_text',x.get('psm6_text',x.get('new_text',x.get('text2',x.get('alternate_psm6','')))))
 if not new:
  print('keys',x.keys());raise RuntimeError('unknown second view key')
 source[x['book']]['original_text'].append(old)
 source[x['book']]['alternative_text'].append(new)
print('source documents', {k:(len(v['original_text']),len(v['alternative_text'])) for k,v in source.items()})
results={}
for book,views in source.items():
 scores={}
 for view,docs in views.items():
  ww=[w for txt in docs for w in forms(txt) if w.isascii() and w.isalpha() and len(w)>1]
  supported=[]
  for w in ww:
   if all(models[m].unk_id() not in models[m].encode(w,out_type=int) for m in models):supported.append(w)
  unseen=[w for w in supported if w not in train]
  s=sum(map(len,unseen))
  scores[view]={'words':len(ww),'unseen':len(unseen), 'per_tokenizer':{f'{mix}-{seed}':100*sum(len(models[(mix,seed)].encode(w,out_type=int)) for w in unseen)/s for mix,seed in models}}
 comparisons=[]
 for pair in [(0,50),(0,100),(50,100)]:
  for seed in [11,23,37,47,59]:
   a,b=pair;d1=scores['original_text']['per_tokenizer'][f'{a}-{seed}']-scores['original_text']['per_tokenizer'][f'{b}-{seed}'];d2=scores['alternative_text']['per_tokenizer'][f'{a}-{seed}']-scores['alternative_text']['per_tokenizer'][f'{b}-{seed}'];comparisons.append({'pair':pair,'seed':seed,'d_original':d1,'d_alt':d2,'flip':d1*d2<0})
 results[book]={'scores':scores,'comparisons':comparisons}
 print(book,'flip_count',sum(x['flip'] for x in comparisons),'of',len(comparisons),'view_words',[(k,v['words']) for k,v in scores.items()])
BASE.joinpath('posthoc_two_view_audit.json').write_text(json.dumps(results,indent=2))
