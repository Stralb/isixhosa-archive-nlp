#!/usr/bin/env python3
"""Independent lightweight verification from saved scores and fixed input snapshots.
Does not generate paper or interpret absence of external genre labels as ground truth.
"""
from pathlib import Path
import collections,hashlib,csv,json,math,os
import numpy as np
D=Path(__file__).parent; P=Path(os.environ.get('PHASE26_DIR',str(D/'inputs'/'phase26')))

def csvload(fp):
 with (D/fp).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))

def checksaved(file):return json.loads((D/file).read_text())

frozen=[l.split(maxsplit=1) for l in (D/'FROZEN_INPUT_SHA256.txt').read_text().splitlines()]
for dig,path in frozen:
 path=Path(path)
 target=(P/'windows.jsonl') if path.name=='windows.jsonl' else D/path.name
 assert hashlib.sha256(target.read_bytes()).hexdigest()==dig,(target,'frozen bytes mismatch')
W=[json.loads(x) for x in (P/'windows.jsonl').read_text().splitlines()]
assert len(W)==581 and len({r['file'] for r in W})==581
byfile={r['file']:r for r in W}
A=checksaved('RESULTS.json');B=checksaved('ORDER_SENSITIVITY_RESULTS.json');C=checksaved('LENGTH_STANDARDIZATION_RESULTS.json')
rows=csvload('all_predictions.csv'); order=csvload('order_sensitivity_predictions.csv');bins=csvload('length_page_bin_counts.csv')
assert A['data']['train_windows']==374 and A['data']['test_windows']==207
assert len(rows)==(207*6+195*5) # six complete-test and five retained-test conditions

def issue_macro(items):
 grp=collections.defaultdict(list)
 for r in items:grp[(r['source'],r['issue'])].append(int(r['correct']))
 return float(np.mean([np.mean([np.mean(vals) for (ss,_),vals in grp.items() if ss==s]) for s in ('IMVO','ISIG')]))

for name,val in A['conditions'].items():
 sel=[r for r in rows if r['condition']==name]
 assert sel and abs(issue_macro(sel)-val['macro'])<1e-13,name
 assert len(sel)==val['n_pages']
 assert all(int(r['pred'])==int(float(r['p_imvo'])>=.5) for r in sel)

for name in ('word1','word12','cross_char'):
 sel=[r for r in order if r['condition']==name and r['mode']=='ordered']
 assert abs(issue_macro(sel)-B['ordered'][name])<1e-13
 for j in range(3):
  sub=[r for r in order if r['condition']==name and r['mode']==f'shuffle_{j}']
  assert abs(issue_macro(sub)-B['shuffled'][name][j])<1e-13
assert len(order)==207*3*4
# Word unigram's feature representation invariant; within-word char-wb also invariant
for condfile,methodname,cases in [('order_sensitivity_predictions.csv','word1',['ordered','shuffle_0','shuffle_1','shuffle_2']),('all_predictions.csv',None,['ordered','shuffle_0','shuffle_1','shuffle_2'])]:
 group=collections.defaultdict(dict)
 for r in csvload(condfile):
  if r.get('condition')==(methodname if methodname else r.get('condition')) and (r['mode'] if methodname else r['condition']) in cases:
   group[r['file']][r['mode'] if methodname else r['condition']]=float(r['p_imvo'])
 assert len(group)==207
 assert all(set(g)==set(cases) for g in group.values())
 assert max(abs(x['ordered']-x[z]) for x in group.values() for z in cases[1:])<1e-12
# Boilerplate repeat labels recover from source windows, without trusting reported flags
n=13;owner=collections.defaultdict(set)
for r in W:
 for i in range(len(r['tokens'])-n+1):owner[tuple(r['tokens'][i:i+n])].add(r['issue'])
repeated={t for t,s in owner.items() if len(s)>1}
assert len(repeated)==A['data']['repeated_13gram_types']
flags={r['file']:any(tuple(r['tokens'][i:i+n]) in repeated for i in range(len(r['tokens'])-n+1)) for r in W}
assert sum(flags.values())==A['data']['repeated_windows_total']
assert all(not flags[r['file']] for r in rows if r['condition']=='no_repeat_ordered')

# Independently reconstruct per-window BPE rates and standardized source gap from per-bin counts
assert len(bins)==207*3*5
past={}
with (P/'tokenizer_page_scores.csv').open(newline='',encoding='utf8') as f:
 for r in csv.DictReader(f):past[(r['file'],int(r['seed']))]=r
agg=collections.defaultdict(lambda:np.zeros(3))
for r in bins:
 k=(r['file'],int(r['seed']));agg[k]+=[float(r['nchars']),float(r['IMVO_pieces']),float(r['ISIG_pieces'])]
for k,a in agg.items():
 orig=past[k]
 assert all(abs(a[i]-float(orig[col]))<1e-12 for i,col in enumerate(['alphabetic_chars','IMVO_pieces','ISIG_pieces']))
share={s:collections.defaultdict(float) for s in ('IMVO','ISIG')}
for r in bins:
 if r['seed']=='0': share[r['source']][r['bin']]+=float(r['nchars'])
lengths=['1-3','4-6','7-9','10-12','13+']
common=[]
for lab in lengths:
 common.append(.5*sum(share[s][lab]/sum(share[s].values()) for s in ('IMVO','ISIG')))
assert abs(sum(common)-1)<1e-13
assert all(abs(wt-C['common_character_mass_weights'][lab])<1e-12 for lab,wt in zip(lengths,common))
for source in ('IMVO','ISIG'):
 vals=[]
 for seed in range(3):
  vals_bin={lab:np.zeros(2) for lab in lengths}
  for r in bins:
   if r['source']!=source or int(r['seed'])!=seed:continue
   diff=float(r['ISIG_pieces'])-float(r['IMVO_pieces'])
   if source=='ISIG':diff=-diff
   vals_bin[r['bin']]+=[float(r['nchars']),diff]
  raw=100*sum(t[1] for t in vals_bin.values())/sum(t[0] for t in vals_bin.values())
  standard=sum(common[i]*100*vals_bin[lab][1]/vals_bin[lab][0] for i,lab in enumerate(lengths))
  vals.append([raw,standard])
 assert abs(np.mean(np.array(vals)[:,0])-C['means'][source]['raw'])<1e-11
 assert abs(np.mean(np.array(vals)[:,1])-C['means'][source]['standardized'])<1e-11
assert C['gates']['C1_residual_positive_both'] is True
assert C['gates']['C2_strong_mediation_both'] is False
assert B['gates']=={'H29B1_bigram_beats_unigram':False,'H29B2_bigram_order_effect':False,'H29B3_cross_char_order_effect':False}
info={'passed':True,'hashes_verified':len(frozen),'n_historical_windows':len(W),'n_prediction_rows_phase29a':len(rows),'n_prediction_rows_phase29b':len(order),'n_bpe_length_bin_rows':len(bins),'n_BPE_page_scores_reproduced':len(agg),'n_reused_13gram_types':len(repeated),'word_unigram_and_charwb_shuffles_exactly_invariant':True,'all_primary_metrics_recomputed':True,'all_length_standardized_means_recomputed':True,'metadata_labels_available':False}
(D/'VERIFICATION.json').write_text(json.dumps(info,indent=2))
print(json.dumps(info,indent=2))
