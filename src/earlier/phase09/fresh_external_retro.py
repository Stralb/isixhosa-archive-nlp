#!/usr/bin/env python3
"""Retrospective external scoring of 50 new frozen branch-rule draws."""
import collections,hashlib,json,sys
from pathlib import Path
import numpy as np
from classic_bpe_intervene import ClassicBPE
import run_phase9 as p
ROOT=p.ROOT;P8=p.p8
assert hashlib.sha256((ROOT/'FRESH_EXTERNAL_RETRO_PROTOCOL.md').read_bytes()).hexdigest()==(ROOT/'FRESH_EXTERNAL_RETRO_PROTOCOL.sha256').read_text().split()[0]
selection=json.loads((ROOT/'fresh_selection_checkpoint.json').read_text())['runs']
assert len(selection)==50
trained_seen={w for pg in P8.training_pages_cached for w in P8.forms(pg['text'])}
corpora={}
for label,fn in [('ityala','ityala_lamawele_raw.txt'),('zem','zemkinkomo_magwalandini_raw.txt')]:
  corpora[label]=[{'text':(Path('/mnt/data/isiXhosa_historical_eval/ocr')/fn).read_text(encoding='utf8')}]
modern=[]
for line in Path('/mnt/data/xho.jsonl').read_text(encoding='utf8').splitlines():
  if not line.strip():continue
  j=json.loads(line)
  if isinstance(j.get('text'),str) and P8.alpha_len(j['text'])>0:modern.append(j)
assert len(modern)==181
corpora['modern']=modern

DATA=ROOT/'fresh_external_data';DATA.mkdir(exist_ok=True)
model_spec=[]
for z in selection:
 for choice in z['choices']:
   model_spec.append((z['source_mix'],z['seed'],choice['option'],choice['model_name']))
assert len(model_spec)==80, len(model_spec)


def precache(start,stop):
 for label,docs in corpora.items():
  words=collections.Counter(w for doc in docs for w in P8.forms(doc['text']))
  lex=list(words)
  for i in range(start,stop):
    mix,seed,op,name=model_spec[i];dest=DATA/f'{label}_{name}.npy'
    if dest.exists():continue
    model=ClassicBPE.from_json(json.loads((ROOT/'fresh_draws'/(name+'.json')).read_text()))
    vec=np.array([model.encode_len(w) for w in lex],dtype=np.int16)
    np.save(dest,vec)
    if (i+1)%10==0:print('EXT_CACHED',label,i+1,'/80',flush=True)
 print('EXT_CHUNK_DONE',start,stop,flush=True)


def analyze():
 rng=np.random.default_rng(20261023);B=10000
 summaries={}
 for label,docs in corpora.items():
    cnt=collections.Counter(w for doc in docs for w in P8.forms(doc['text']))
    words=list(cnt)
    freq=np.array([cnt[w] for w in words],dtype=np.int64)
    alpha=np.array([P8.alpha_len(w) for w in words],dtype=np.int64)
    unseen=np.array([w not in trained_seen for w in words],dtype=bool)
    enc=[]
    for mix,seed,op,name in model_spec:
      dest=DATA/f'{label}_{name}.npy';assert dest.exists(),dest
      e=np.load(dest);assert len(e)==len(words)
      enc.append(e)
    good=np.logical_and.reduce([e!=-1 for e in enc])
    raw=[]
    for (mix,seed,op,name),vec in zip(model_spec,enc):
       record={'source_mix':mix,'seed':seed,'option':op,'model_name':name}
       for tag,mask in [('unseen',good&unseen),('all',good)]:
           den=int(np.sum(freq[mask]*alpha[mask])); assert den>0
           record[tag+'_E']=100*int(np.sum(freq[mask]*vec[mask]))/den
       raw.append(record)
    by={(v['source_mix'],v['seed'],v['option']):v for v in raw}
    paired=[];bal=[]
    for row in selection:
      mix=row['source_mix'];seed=row['seed']
      nat=by[mix,seed,'greedy'];sel=by[mix,seed,row['selected_option']]
      if mix==1.:
        paired.append({'seed':seed,'selected_E':sel['unseen_E'],'natural_E':nat['unseen_E'],
             'difference':sel['unseen_E']-nat['unseen_E'],
             'all_difference':sel['all_E']-nat['all_E']})
      else:bal.append(nat['unseen_E'])
    arr=np.array([r['difference'] for r in paired]);idx=rng.integers(0,30,size=(B,30))
    bd=np.mean(arr[idx],axis=1)
    alls=np.array([r['all_difference'] for r in paired]);
    selected=np.array([r['selected_E'] for r in paired]);natural=np.array([r['natural_E'] for r in paired]);
    summary={'n_docs':len(docs),'total_word_occurrences':int(sum(freq)),
      'unseen_occurrences':int(np.sum(freq[unseen])),'common_supported_occurrences':int(np.sum(freq[good])),
      'common_supported_unseen_occurrences':int(np.sum(freq[good&unseen])),
      'support_percentage_all':float(100*np.sum(freq[good])/sum(freq)),
      'random_natural_mean_E':float(natural.mean()),
      'random_natural_sd_E':float(natural.std(ddof=1)),
      'training_only_selected_mean_E':float(selected.mean()),
      'training_only_selected_sd_E':float(selected.std(ddof=1)),
      'balanced_mean_E':float(np.mean(bal)),
      'selected_minus_natural_mean_E':float(arr.mean()),
      'paired_bootstrap_95':np.quantile(bd,[.025,.975]).tolist(),
      'selected_minus_natural_all_E':float(alls.mean()),
      'selected_wins':int(np.sum(arr< -1e-8)),'selected_worse':int(np.sum(arr>1e-8)),
      'raw_paired':paired}
    summaries[label]=summary
    print('EXTERNAL',label,'selected',round(summary['training_only_selected_mean_E'],4),
      'random',round(summary['random_natural_mean_E'],4),'bal',round(summary['balanced_mean_E'],4),
      'delta',round(summary['selected_minus_natural_mean_E'],4),'ci',summary['paired_bootstrap_95'],flush=True)
 overall=all(summaries[c]['selected_minus_natural_mean_E']<0 and summaries[c]['paired_bootstrap_95'][1]<0 for c in summaries)
 out={'status':'retrospective previously inspected external corpora',
  'protocol_sha256':hashlib.sha256((ROOT/'FRESH_EXTERNAL_RETRO_PROTOCOL.md').read_bytes()).hexdigest(),
  'frozen_selection_sha256':hashlib.sha256((ROOT/'fresh_selection_checkpoint.json').read_bytes()).hexdigest(),
  'retrospective_transfer_gate':overall,'corpora':summaries}
 (ROOT/'fresh_external_retro_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
 print('TRANSFER_GATE',overall,flush=True)

if __name__=='__main__':
 if sys.argv[1]=='cache':precache(int(sys.argv[2]),int(sys.argv[3]))
 elif sys.argv[1]=='analyze':analyze()
 else:raise ValueError(sys.argv[1])
