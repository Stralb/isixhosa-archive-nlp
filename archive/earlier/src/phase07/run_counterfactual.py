#!/usr/bin/env python3
"""Frozen OCR-proxy perturbation; diagnostic, not historical OCR ground truth."""
import collections, functools, hashlib, json, random, statistics, sys, time, zlib
from pathlib import Path
import sentencepiece as spm
ROOT=Path(__file__).resolve().parent
EX=Path('/mnt/data/isiXhosa_historical_eval/isiXhosa_external_validation')
sys.path.insert(0,str(EX))
from common import dev_pages,training_pages,forms,alpha_len
mixes=[0,0.5,1]
seeds=[11,23,37,47,59]
manifest=json.loads((EX/'models/training_manifest.json').read_text())
assert manifest['seeds']==seeds and manifest['mixes']==mixes
selection=json.loads((EX/'models/selection_frozen.json').read_text())
assert selection['raw_parity_selected']==0.5 and selection['opportunity_selected']==1
models={}
for rr in manifest['runs']:
    path=EX/'models'/(rr['name']+'.model')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==rr['model_sha256'], path
    models[(rr['imvo_share'],rr['seed'])]=spm.SentencePieceProcessor(model_file=str(path))
model_list=[models[(mix,seed)] for mix in mixes for seed in seeds]
alphabet=set(manifest['alphabet'])
training=training_pages()
seen={w for p in training for w in forms(p['text'])}
original_pages=[{'id':p['id'],'source':p['source'],'words':forms(p['text'])} for p in dev_pages()]
assert len(original_pages)==72 and len(training)==218
assert collections.Counter(p['source'] for p in original_pages)=={'IMVO':48,'ISIG':24}
# Glyph-confusion proxy only; pairs NOT fitted or calibrated to a real OCR engine.
CONF={'a':'o','b':'h','c':'e','d':'b','e':'c','f':'s','g':'q','h':'b','i':'l','j':'i',
      'k':'h','l':'i','m':'n','n':'u','o':'c','p':'q','q':'p','r':'n','s':'f','t':'l','u':'n','v':'y','w':'v','x':'k','y':'v','z':'s'}

def eligible_target(w):
    return len(w)>=5 and w.isascii() and w.isalpha()

@functools.lru_cache(maxsize=None)
def cost(w):
    if not w or any(c.isalpha() and c not in alphabet for c in w):return None
    arr=[]
    for model in model_list:
        ids=model.encode(w,out_type=int)
        if model.unk_id() in ids:return None
        arr.append(len(ids))
    return tuple(arr)

def mutations(word,rng):
    idx=rng.randrange(len(word))
    random_new=chr(97+(ord(word[idx])-97+1+rng.randrange(25))%26)
    splitpos=rng.randrange(2,len(word)-1)
    return (word[:idx]+CONF[word[idx]]+word[idx+1:],
            word[:idx]+random_new+word[idx+1:],
            (word[:splitpos],word[splitpos:]))

def build_draws(rep):
    all_pages=[]
    for p in original_pages:
        rng=random.Random(20261023+1000003*rep+zlib.crc32(p['id'].encode()))
        draws=[]
        for w in p['words']:
            if eligible_target(w):draws.append((rng.random(),mutations(w,rng)))
            else:draws.append(None)
        all_pages.append((p,draws))
    return all_pages

def evaluate(draws,rate,mode):
    cts={s:collections.Counter() for s in ('IMVO','ISIG')}
    false={s:collections.Counter() for s in ('IMVO','ISIG')}
    originals={s:0 for s in ('IMVO','ISIG')}
    eligible={s:0 for s in ('IMVO','ISIG')}
    altered={s:0 for s in ('IMVO','ISIG')}
    for p,ds in draws:
        group=p['source'];originals[group]+=len(ds)
        for w,d in zip(p['words'],ds):
            if d is not None:eligible[group]+=1
            if d is not None and d[0]<rate:
                altered[group]+=1
                changed=d[1][mode]
                obs=(changed,) if isinstance(changed,str) else changed
            else:obs=(w,)
            cts[group].update(obs)
            if w in seen:false[group].update(x for x in obs if x not in seen)
    result={}
    for group in ('IMVO','ISIG'):
        c=cts[group];f=false[group]
        total=sum(c.values());unseen=sum(n for w,n in c.items() if w not in seen)
        art=sum(f.values());sup_all=[0]*15;sup_unseen=[0]*15
        denom_all=0;denom_unseen=0;sup_occ=0
        for w,n in c.items():
            costs=cost(w)
            if costs is None:continue
            chars=n*alpha_len(w);denom_all+=chars;sup_occ+=n
            for k,v in enumerate(costs):sup_all[k]+=v*n
            if w not in seen:
                denom_unseen+=chars
                for k,v in enumerate(costs):sup_unseen[k]+=v*n
        assert denom_all>0 and denom_unseen>0
        result[group]={'source_word_occurrences':originals[group],
                       'observed_word_occurrences':total,'eligible_original_occs':eligible[group],
                       'actual_changed_occs':altered[group],'observed_unseen_occs':unseen,
                       'observed_unseen_fraction':unseen/total,'artificially_unseen_occs':art,
                       'artificial_share_of_observed_unseen':art/unseen,
                       'scored_alpha_all':denom_all,'scored_alpha_unseen':denom_unseen,
                       'common_tokenizer_supported_occ_fraction':sup_occ/total,
                       'cost_all_by_mix_seed':{str(mix):[100*sup_all[5*i+j]/denom_all for j in range(5)] for i,mix in enumerate(mixes)},
                       'cost_unseen_by_mix_seed':{str(mix):[100*sup_unseen[5*i+j]/denom_unseen for j in range(5)] for i,mix in enumerate(mixes)}}
        p=result[group]['cost_unseen_by_mix_seed']
        result[group]['parity_minus_opportunity_by_seed']=[x-y for x,y in zip(p['0.5'],p['1'])]
        result[group]['parity_minus_opportunity_mean']=statistics.mean(result[group]['parity_minus_opportunity_by_seed'])
    return result

def main():
    start=time.time()
    print('train forms',len(seen),'validation pages',len(original_pages),'word occurrences',sum(len(p['words']) for p in original_pages),flush=True)
    base=evaluate(build_draws(0),0,0)
    print('baseline',[(d,base[d]['observed_unseen_fraction'],base[d]['parity_minus_opportunity_mean']) for d in ('IMVO','ISIG')],flush=True)
    rows=[]
    for rep in range(20):
        d=build_draws(rep)
        for mode,name in enumerate(('confusable','random','split')):
            for rate in (0.01,0.03,0.05,0.10):
                rows.append({'replicate':rep,'mode':name,'target_fraction':rate,'by_source':evaluate(d,rate,mode)})
        if (rep+1)%2==0:
            print('replicate',rep+1,'/',20,'elapsed',round(time.time()-start,2),'cache',cost.cache_info(),flush=True)
            (ROOT/'results_checkpoint.json').write_text(json.dumps({'baseline':base,'rows':rows},separators=(',',':')))
    result={'scope':'retrospective synthetic stress test on 1887 held-out newspaper validation; no true OCR labels or independent confirmation',
            'precommit_sha256':hashlib.sha256((ROOT/'PRECOMMIT.md').read_bytes()).hexdigest(),
            'frozen_selection_sha256':hashlib.sha256((EX/'models/selection_frozen.json').read_bytes()).hexdigest(),
            'baseline':base,'rows':rows,'cache':str(cost.cache_info()),'seconds':time.time()-start}
    (ROOT/'simulation_results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print('done',len(rows),'seconds',result['seconds'],flush=True)

if __name__=='__main__':main()
