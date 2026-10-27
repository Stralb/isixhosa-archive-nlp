#!/usr/bin/env python3
"""Phase8H: historical book OCR cleaning, overlap, and UNK rate robustness."""
import collections, json, re, sys, time
from pathlib import Path
import numpy as np
import sentencepiece as spm
import run_phase8 as r
ST=time.monotonic(); R=r.ROOT; OCR=Path('/mnt/data/isiXhosa_historical_eval/ocr')
q=json.loads((R/'selfscore_results.json').read_text()); new=q['new_models']
first=[new[3*i] for i in range(20)]
selected=[min(new[3*i:3*i+3],key=lambda z:(z['self_train_E'],z['seed'])) for i in range(20)]
bal=[z for z in q['baseline_models'] if z['mix']==.5]
large=json.loads((R/'triplemass_training_checkpoint.json').read_text())['runs']
full=json.loads((R/'large_single_training.json').read_text())['runs'][1]
model_spec=[(k,z['seed'],str(r.MODELS/(z['name']+'.model'))) for k,ls in [('selected',selected),('first',first),('balanced',bal)] for z in ls]
model_spec += [('large_single',z['seed'],z['model_file']) for z in large]
model_spec += [('full_pool',0,full['model_file'])]
assert len(model_spec)==81
from hashlib import sha256
shingles=set()
for rec in r.training_pages_cached:
    for paragraph in re.split(r'\n\s*\n+',rec['text']):
        tokens=r.forms(paragraph)
        shingles.update(tuple(tokens[i:i+12]) for i in range(len(tokens)-11))
seen={w for p in r.training_pages_cached for w in r.forms(p['text'])}
alphabet={c for w in seen for c in w if c.isalpha()}
def cleaner(text,join=False):
    text=re.sub(r'(?mi)^\s*University of Fort Hare\s*$','',text)
    text=re.sub(r'(?mi)^\s*Together in Excellence\s*$','',text)
    if join:text=re.sub(r'(?<=[a-zA-Z])-\s*\n\s*(?=[a-zA-Z])','',text)
    return text

def get_counts(book,variant):
    join='joined' in variant
    exclude='exclude' in variant
    f=OCR/(book+'_pages.jsonl')
    data=[json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    cnt=collections.Counter(); excluded=0; total=0
    for p in data:
        text=cleaner(p['text'],join)
        for par in re.split(r'\n\s*\n+',text):
            words=r.forms(par); total+=len(words)
            covered=set()
            if exclude:
                for j in range(len(words)-11):
                    if tuple(words[j:j+12]) in shingles: covered.update(range(j,j+12))
            excluded+=len(covered)
            cnt.update(w for i,w in enumerate(words) if i not in covered)
    return cnt,{'pages_and_sides':len(data),'original_word_occurrences':total,'excluded_word_occurrences':excluded}

def score(book,variant):
    cnt,meta=get_counts(book,variant);unique=list(cnt)
    weights=np.array([cnt[w] for w in unique],dtype=np.int64)
    lengths=np.array([r.alpha_len(w) for w in unique],dtype=np.int64)
    unseen=np.array([w not in seen for w in unique],dtype=bool)
    allowed=np.array([all(not c.isalpha() or c in alphabet for c in w) for w in unique],dtype=bool)
    encoded=[];unknowns=[]
    for label,seed,path in model_spec:
        p=spm.SentencePieceProcessor(model_file=path)
        v=p.encode(unique,out_type=int,num_threads=4)
        code=np.array([len(x) if p.unk_id() not in x else -1 for x in v],dtype=np.int32)
        unknowns.append(int(np.sum(weights[code==-1])))
        allowed&=(code!=-1)
        encoded.append(code)
    common=allowed;novel=common&unseen
    assert np.any(novel)
    data={'meta':meta,'n_word_occurrences':int(np.sum(weights)),
        'common_supported_word_occurrences':int(np.sum(weights[common])),
        'common_supported_fraction':float(np.sum(weights[common])/np.sum(weights)),
        'common_supported_unseen_word_occurrences':int(np.sum(weights[novel])),
        'per_policy':{},'selected_minus_first':{},'all_models':[]}
    for i,((name,seed,path),v) in enumerate(zip(model_spec,encoded)):
        z={'policy':name,'seed':seed,'unk_word_occurrences':unknowns[i],
           'unk_word_fraction':unknowns[i]/sum(weights)}
        for label,mask in [('unseen',novel),('all',common)]:
            n=int(np.sum(weights[mask]*v[mask]));d=int(np.sum(weights[mask]*lengths[mask]));z[label+'_E']=100*n/d
        data['all_models'].append(z)
    for name in ['selected','first','balanced','large_single','full_pool']:
        a=[x for x in data['all_models'] if x['policy']==name]
        data['per_policy'][name]={'n_models':len(a),
        'unseen_mean':float(np.mean([x['unseen_E'] for x in a])),
        'unseen_q90':float(np.quantile([x['unseen_E'] for x in a],.9)) if len(a)>1 else None,
        'all_mean':float(np.mean([x['all_E'] for x in a])),
        'unk_word_fraction_mean':float(np.mean([x['unk_word_fraction'] for x in a])),
        'unk_word_fraction_range':[float(min(x['unk_word_fraction'] for x in a)),float(max(x['unk_word_fraction'] for x in a))]}
    sel=np.array([z['unseen_E'] for z in data['all_models'] if z['policy']=='selected'])
    fir=np.array([z['unseen_E'] for z in data['all_models'] if z['policy']=='first'])
    inds=np.random.default_rng(20261029).integers(0,20,size=(10000,20))
    diffs=(sel-fir)[inds].mean(axis=1)
    data['selected_minus_first']={'unseen_mean_delta':float(np.mean(sel-fir)),
        'paired_bootstrap95':np.quantile(diffs,[.025,.975]).tolist(),
        'selected_wins':int(np.sum(sel<fir)), 'selected_equal':int(np.sum(sel==fir))}
    data['selected_minus_full']={'unseen_mean_delta':float(np.mean(sel)-data['per_policy']['full_pool']['unseen_mean'])}
    return data

out={'protocol':'EXTERNAL_QA_PROTOCOL.md','train_shingles':len(shingles),'corpora':{},'note':'Retrospective, fixed models and OCR variants, no new independent confirmation'}
for book in ('ityala_lamawele','zemkinkomo_magwalandini'):
    out['corpora'][book]={}
    for variant in ('clean_raw','clean_joined','clean_raw_exclude','clean_joined_exclude'):
        a=score(book,variant)
        out['corpora'][book][variant]=a
        print(book,variant,'n',a['n_word_occurrences'],'excluded',a['meta']['excluded_word_occurrences'],
          'support',round(a['common_supported_fraction'],4),
          'selected-first',round(a['selected_minus_first']['unseen_mean_delta'],4),
          'selected-full',round(a['selected_minus_full']['unseen_mean_delta'],4),flush=True)
    assert out['corpora'][book]['clean_raw']['meta']['excluded_word_occurrences']==0
assert out['corpora']['ityala_lamawele']['clean_raw']['n_word_occurrences']==29215
assert out['corpora']['zemkinkomo_magwalandini']['clean_raw']['n_word_occurrences']==56563
path=R/'external_qa_results.json'
out['runtime_seconds']=time.monotonic()-ST
path.write_text(json.dumps(out,indent=2))
print('DONE',path,round(out['runtime_seconds'],2),flush=True)
