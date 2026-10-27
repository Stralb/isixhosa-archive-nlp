"""Post-frozen descriptive/robustness audit for supplied modern xho.jsonl.
No tokenizer retraining, no changed model-selection rules, no downstream scores.
"""
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json
import os
import re
import numpy as np
import sentencepiece as spm
from common import training_pages, forms, alpha_len

BASE = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get('MODERN_CORPUS_PATH', BASE / 'external_raw' / 'vukuzenzele_xho' / 'xho.jsonl'))
SCORE = BASE / 'modern_external_scores_frozen.json'
MODELS = BASE / 'models'
OUTPUT = BASE / 'modern_external_audit.json'

source_bytes = SOURCE.read_bytes()
source_rows = [json.loads(line) for line in source_bytes.decode('utf-8').splitlines() if line.strip()]
scored = json.loads(SCORE.read_text())
sel = json.loads((MODELS / 'selection_frozen.json').read_text())
manifest = json.loads((MODELS / 'training_manifest.json').read_text())
assert sel['raw_parity_selected'] == 0.5 and sel['opportunity_selected'] == 1
assert hashlib.sha1(f'blob {len(source_bytes)}\0'.encode() + source_bytes).hexdigest() == '99740ae0b41fa12e0bf7cee686391cabff72ff1f'
assert set(str(r.get('language_code')) for r in source_rows) == {'xho'}
for rec in manifest['runs']:
    content = (MODELS / (rec['name']+'.model')).read_bytes()
    assert hashlib.sha256(content).hexdigest() == rec['model_sha256']
models = {(x['imvo_share'],x['seed']): spm.SentencePieceProcessor(model_file=str(MODELS/(x['name']+'.model'))) for x in manifest['runs']}
train_forms = {w for p in training_pages(os.environ.get('NEWSPAPER_ZIP_PATH', '/mnt/data/TxtLangFiles1870-1894_TB.zip')) for w in forms(p['text'])}
alphabet = set(manifest['alphabet'])
report_rows = scored['document_results']['vukuzenzele_xho']
assert len(source_rows) == 186 and len(report_rows) == 181
seeds = manifest['seeds']
analysis=[]
word_counter = Counter()
novel_counter = Counter()
for record in report_rows:
    match=re.search(r'article_(\d{4})_',record['id'])
    assert match is not None
    i=int(match.group(1))
    src=source_rows[i]
    text=src['text']; words=forms(text)
    assert hashlib.sha256(text.encode('utf-8')).hexdigest()==record['sha256_text']
    eligible={}
    for w in set(words):
        eligible[w] = all((not c.isalpha() or c in alphabet) for c in w) and all(model.unk_id() not in model.encode(w,out_type=int) for model in models.values())
    full=[w for w in words if eligible[w]]
    unseen=[w for w in full if w not in train_forms]
    full_alpha=sum(map(alpha_len,full)); unseen_alpha=sum(map(alpha_len,unseen))
    assert len(full)==record['common_tokenizer_supported_word_occurrences']
    assert len(words)==record['words'] and sum(map(alpha_len,words))==record['alpha_chars']
    for w in words: word_counter[w]+=1
    for w in unseen: novel_counter[w]+=1
    def ecost(mix,metric):
        r=next(c for c in record['costs'] if c['imvo_share']==mix)
        key='supported_unseen_E_by_seed' if metric=='unseen' else 'supported_E_by_seed'
        return np.asarray(r[key],dtype=float)
    diff_unseen=ecost(1,'unseen')-ecost(.5,'unseen')
    diff_all=ecost(1,'all')-ecost(.5,'all')
    analysis.append({'source_index':i,'edition':src['edition'].strip(), 'alpha_chars_supported':full_alpha,
       'unseen_alpha_chars_supported':unseen_alpha,'words':len(words), 'excluded_words':len(words)-len(full),
       'unseen_words_supported':len(unseen),'delta_unseen_by_seed':diff_unseen.tolist(),
       'delta_all_by_seed':diff_all.tolist(),
       'unseen_by_seed':{str(m):ecost(m,'unseen').tolist() for m in manifest['mixes']},
       'all_by_seed':{str(m):ecost(m,'all').tolist() for m in manifest['mixes']}})

# Resample full issues/editions rather than pretending all articles are independent.
rng=np.random.default_rng(20261022)
byed=defaultdict(list)
for row in analysis: byed[row['edition']].append(row)
ed_keys=list(sorted(byed))

def summary(metric,weighting='article',nbootstrap=12000):
    denominator_key='unseen_alpha_chars_supported' if metric=='unseen' else 'alpha_chars_supported'
    use='delta_unseen_by_seed' if metric=='unseen' else 'delta_all_by_seed'
    group_num=[];group_den=[]
    all_delta=[];all_weight=[];all_ed=[]
    for e in ed_keys:
        ww=[];dd=[]
        for x in byed[e]:
            wt=x[denominator_key] if weighting=='chars' else 1
            if wt==0:continue
            ww.append(wt);dd.append(np.array(x[use],dtype=float))
            all_weight.append(wt);all_delta.append(np.array(x[use],dtype=float));all_ed.append(e)
        group_num.append(np.sum(np.array(dd)*np.array(ww)[:,None],axis=0) if dd else np.zeros(len(seeds)))
        group_den.append(sum(ww))
    gnum=np.array(group_num);gden=np.array(group_den)
    true_d=gnum.sum(axis=0)/gden.sum()
    sample=rng.integers(0,len(ed_keys),size=(nbootstrap,len(ed_keys)))
    bsnum=gnum[sample].sum(axis=(1,2))
    bsden=gden[sample].sum(axis=1)*len(seeds)
    bs=bsnum/bsden
    perarticle=np.array(all_delta)
    return {'delta_opportunity_minus_parity':float(np.mean(true_d)),
            'difference_by_seed':[float(x) for x in true_d],
            'bootstrap_edition_95_interval':[float(x) for x in np.quantile(bs,[0.025,0.975])],
            'article_win_fraction_on_seed_mean':float(np.mean(perarticle.mean(axis=1)<0)),
            'fraction_seeds_with_improvement':float(np.mean(true_d<0)),
            'n_articles':len(perarticle),'n_editions':len(ed_keys),
            'bootstrap_units':'editions, resampled with replacement; five seeds averaged within each sample',
            'weights':weighting}

def corpus_means(metric):
    denom='unseen_alpha_chars_supported' if metric=='unseen' else 'alpha_chars_supported'
    cost='unseen_by_seed' if metric=='unseen' else 'all_by_seed'
    result={}
    for mix in manifest['mixes']:
        sums=np.zeros(len(seeds)); wts=0
        for row in analysis:
            wt=row[denom];sums+=np.array(row[cost][str(mix)])*wt
            wts+=wt
        result[str(mix)]={'corpus_char_weighted_E':float(np.mean(sums/wts)),
                          'by_seed':[float(x) for x in sums/wts]}
    return result

# Post-hoc OCR/segmentation sensitivity: score words without internal hyphens or apostrophes.
# Uses same frozen models and union-train unseen definition; not a new selection experiment.
ocr_rows=[]
for row in analysis:
    idx=row['source_index']; words=forms(source_rows[idx]['text'])
    eligible=[w for w in words if w.isalpha() and all(c in alphabet for c in w)]
    eligible=[w for w in eligible if all(m.unk_id() not in m.encode(w,out_type=int) for m in models.values())]
    novel=[w for w in eligible if w not in train_forms]
    weights=sum(map(alpha_len,novel))
    if not weights:continue
    deltas=[]
    for seed in seeds:
        cand=models[(1,seed)];parity=models[(.5,seed)]
        deltas.append(100*(sum(len(cand.encode(w,out_type=int)) for w in novel)-sum(len(parity.encode(w,out_type=int)) for w in novel))/weights)
    ocr_rows.append({'edition':source_rows[idx]['edition'].strip(),'chars':weights,'delta_unseen':float(np.mean(deltas)),
                     'by_seed':deltas})

# OCR stratified sensitivity: edition-level cluster bootstrap for alphabetic-only novel-form chars.
ocr_groups=defaultdict(list)
for r in ocr_rows:ocr_groups[r['edition']].append(r)
ocr_keys=sorted(ocr_groups)
ocr_gnum=np.array([sum(sum(v['by_seed'])*v['chars'] for v in ocr_groups[e]) for e in ocr_keys]);ocr_gden=np.array([sum(v['chars'] for v in ocr_groups[e]) for e in ocr_keys]);N=15000
ix=rng.integers(0,len(ocr_keys),size=(N,len(ocr_keys)))
ocr_mean=float(ocr_gnum.sum()/(len(seeds)*ocr_gden.sum()))
ocr_bs=ocr_gnum[ix].sum(axis=1)/(len(seeds)*ocr_gden[ix].sum(axis=1))

result={
 'status':'FROZEN MODERN EXTERNAL SCORING + POSTHOC ROBUSTNESS; not downstream LM evidence',
 'source':{'name':SOURCE.name,'sha256':hashlib.sha256(source_bytes).hexdigest(),'verified_git_blob_sha1':'99740ae0b41fa12e0bf7cee686391cabff72ff1f',
           'records':len(source_rows),'nonempty_scored_articles':len(analysis),'editions':len(ed_keys)},
 'selection':{'parity':sel['raw_parity_selected'],'opportunity':sel['opportunity_selected']},
 'coverage':{'all_surface_word_occurrences':sum(row['words'] for row in analysis),
      'unsupported_surface_word_occurrences':sum(row['excluded_words'] for row in analysis),
      'unseen_surface_word_occurrences_supported':sum(row['unseen_words_supported'] for row in analysis),
      'supported_alpha_chars':sum(row['alpha_chars_supported'] for row in analysis),
      'supported_unseen_alpha_chars':sum(row['unseen_alpha_chars_supported'] for row in analysis),
      'distinct_surface_forms':len(word_counter),
      'distinct_surface_forms_not_in_newspaper_train':sum(w not in train_forms for w in word_counter)},
 'main':{'reported_article_weighted_from_original_frozen_script':scored['external_summary']['vukuzenzele_xho']['model_costs'],
      'corpus_char_weighted':{'all':corpus_means('all'),'unseen':corpus_means('unseen')},
      'paired_differences':{'unseen_article_weighted':summary('unseen','article'),
                            'all_article_weighted':summary('all','article'),
                            'unseen_character_weighted':summary('unseen','chars'),
                            'all_character_weighted':summary('all','chars')}},
 'posthoc_sensitivity':{'excluding_surface_forms_with_internal_punctuation':{'unseen_character_weighted_opportunity_minus_parity':ocr_mean,
      'edition_bootstrap_95_interval':[float(x) for x in np.quantile(ocr_bs,[0.025,0.975])],
      'status':'EXPLORATORY, not a pre-registered primary outcome'}},
 'method_notes':['The five tokenizer seeds share the same newspaper texts; confidence intervals resample 83 observed magazine editions, not hypothetical independent languages.',
                 'No exact within-paragraph 12-word matches were found; this does not exclude shorter or approximate textual reuse.',
                 'Novel forms are exact lowercased surface forms absent from 1887 train newspapers, not semantically novel words.',
                 'Modern magazine text has substantial layout/OCR artifacts; modern transfer is not evidence of historical-book transfer.',
                 'Only tokenization cost is evaluated, not model predictive accuracy or fairness.']
}
OUTPUT.write_text(json.dumps(result,ensure_ascii=False,indent=2))
print('Saved',OUTPUT)
print(json.dumps({'coverage':result['coverage'],'paired_differences':result['main']['paired_differences'],
                  'corpus_char_weighted':result['main']['corpus_char_weighted'],
                  'posthoc_sensitivity':result['posthoc_sensitivity']},indent=2))
