#!/usr/bin/env python3
"""Resumable phase9 scorer; same metrics as protocol and original script."""
import collections,hashlib,json,sys,os
from pathlib import Path
import numpy as np
import sentencepiece as spm
from scipy.stats import spearmanr,pearsonr
from sklearn.mixture import GaussianMixture
from classic_bpe import ClassicBPE
import run_phase9 as phase9
p8=phase9.p8
ROOT=phase9.ROOT
CP=ROOT/'training_checkpoint.json'
RUNS=json.loads(CP.read_text())['runs'];assert len(RUNS)==40
CACHED=ROOT/'encodings';CACHED.mkdir(exist_ok=True)
train=phase9.p8.training_pages_cached
seen={w for p in train for w in p8.forms(p['text'])}
raw={s:collections.Counter(w for p in p8.dev_pages() if p['source']==s for w in p8.forms(p['text'])) for s in ('IMVO','ISIG')}
lex={s:list(raw[s]) for s in raw}
counts={s:np.array([raw[s][w] for w in lex[s]],dtype=np.int64) for s in raw}
lengths={s:np.array([p8.alpha_len(w) for w in lex[s]],dtype=np.int64) for s in raw}
unseen={s:np.array([w not in seen for w in lex[s]],dtype=bool) for s in raw}


def go_chunk(start,stop):
    for i in range(start,stop):
        row=RUNS[i];key=row['model_name'];dest=CACHED/(key+'.npz')
        if dest.exists():print('ALREADY',i+1,key,flush=True);continue
        model=ClassicBPE.from_json(json.loads((phase9.MODELS/(key+'.json')).read_text()))
        sp=spm.SentencePieceProcessor(model_file=str(p8.MODELS/(f"word_uniform_mix{int(row['mix']*100):03d}_draw{row['seed']}.model")))
        out={}
        for d in ('IMVO','ISIG'):
            out[d+'_classic']=np.array([model.encode_len(w) for w in lex[d]],dtype=np.int16)
            es=sp.encode(lex[d],out_type=int,num_threads=4)
            out[d+'_sp']=np.array([len(v) if sp.unk_id() not in v else -1 for v in es],dtype=np.int16)
        np.savez_compressed(dest,**out)
        # verification against direct pieces (not only score lengths)
        for d in ('IMVO','ISIG'):
            for j in (0,len(lex[d])//3,2*len(lex[d])//3,len(lex[d])-1):
                assert out[d+'_classic'][j]==model.encode_len(lex[d][j])
                x=sp.encode(lex[d][j],out_type=int)
                assert out[d+'_sp'][j]==(-1 if sp.unk_id() in x else len(x))
        print('SCORED_CACHED',i+1,'/',len(RUNS),key,flush=True)


def finish():
    cache=[]
    for row in RUNS:
        f=CACHED/(row['model_name']+'.npz');assert f.is_file(),f
        with np.load(f) as z: cache.append({k:z[k] for k in z.files})
    valid={s:np.logical_and.reduce([z[s+'_classic']!=-1 for z in cache]+[z[s+'_sp']!=-1 for z in cache]) for s in raw}
    support={}
    for s in raw:
        support[s]={'valid_occurrences':int(counts[s][valid[s]].sum()),'all_occurrences':int(counts[s].sum()),
            'valid_unseen_occurrences':int(counts[s][valid[s]&unseen[s]].sum()),
            'unseen_occurrences':int(counts[s][unseen[s]].sum()),
            'valid_alpha':int((lengths[s][valid[s]]*counts[s][valid[s]]).sum()),
            'valid_unseen_alpha':int((lengths[s][valid[s]&unseen[s]]*counts[s][valid[s]&unseen[s]]).sum())}
        assert support[s]['valid_unseen_alpha']>0
    rows=json.loads(CP.read_text())['runs']
    for i,row in enumerate(rows):
        row['validation']={}
        for s in raw:
            row['validation'][s]={}
            for name,mask in [('unseen',valid[s]&unseen[s]),('all',valid[s])]:
                denom=int((lengths[s][mask]*counts[s][mask]).sum())
                for short,label in [('classic',''),('sp','sp_')]:
                    numerator=int((cache[i][s+'_'+short][mask]*counts[s][mask]).sum())
                    row['validation'][s][label+name+'_E']=100*numerator/denom
        row['validation']['worst_unseen_E']=max(row['validation'][s]['unseen_E'] for s in raw)
        row['validation']['sp_worst_unseen_E']=max(row['validation'][s]['sp_unseen_E'] for s in raw)
    groups={mi:sorted([r for r in rows if r['mix']==mi],key=lambda x:x['seed']) for mi in (.5,1.)}
    summary={'protocol_sha256':hashlib.sha256((ROOT/'PRECOMMIT.md').read_bytes()).hexdigest(),
        'implementation_sha256':hashlib.sha256((ROOT/'classic_bpe.py').read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(p8.ORIGINAL_ZIP).read_bytes()).hexdigest() if hasattr(p8,'ORIGINAL_ZIP') else hashlib.sha256(Path('/mnt/data/TxtLangFiles1870-1894_TB.zip').read_bytes()).hexdigest(),
        'support':support,'functions':{}}
    for mix in (.5,1.):
        a=np.array([r['validation']['worst_unseen_E'] for r in groups[mix]])
        b=np.array([r['validation']['sp_worst_unseen_E'] for r in groups[mix]])
        gm1=GaussianMixture(n_components=1,random_state=0,n_init=20,reg_covar=1e-6).fit(a.reshape(-1,1))
        gm2=GaussianMixture(n_components=2,random_state=0,n_init=20,reg_covar=1e-6).fit(a.reshape(-1,1))
        summary['functions'][f'mix_{int(mix*100)}']={'classic':{'mean':float(a.mean()),'std':float(a.std(ddof=1)),
           'min':float(a.min()),'max':float(a.max()),'values':a.tolist(),
           'bic_1_minus_bic_2':float(gm1.bic(a.reshape(-1,1))-gm2.bic(a.reshape(-1,1))),
           'mixture_2_centers':sorted(gm2.means_.ravel().tolist())},
           'sentencepiece':{'mean':float(b.mean()),'std':float(b.std(ddof=1)),'values_same_support':b.tolist()},
           'matched_pearson':float(pearsonr(a,b).statistic),
           'matched_spearman':float(spearmanr(a,b).statistic)}
    one=groups[1.];a=np.array([r['validation']['worst_unseen_E'] for r in one]);b=np.array([r['validation']['sp_worst_unseen_E'] for r in one])
    spref=json.loads(p8.OUT.read_text())['runs']
    org={(r['mix'],r['seed']):r['validation']['worst_unseen_E'] for r in spref if r['method']=='word_uniform'}
    old=np.array([org[1.,r['seed']] for r in one]);labels=old>=39.5
    ratio=float(summary['functions']['mix_100']['classic']['std']/summary['functions']['mix_50']['classic']['std'])
    dif=float(a[labels].mean()-a[~labels].mean())
    B=10000;rand=np.random.default_rng(20261023);inds=rand.integers(0,20,size=(B,20))
    corrs=[];differences=[]
    for ix in inds:
        ca=a[ix];cb=b[ix];l=labels[ix]
        corrs.append(spearmanr(ca,cb).statistic if len(set(ca))>1 and len(set(cb))>1 else float('nan'))
        differences.append(ca[l].mean()-ca[~l].mean() if 0<sum(l)<20 else float('nan'))
    b50=np.array([r['validation']['worst_unseen_E'] for r in groups[.5]])
    halfinds=rand.integers(0,20,size=(B,20))
    sdboot=np.std(a[inds],axis=1,ddof=1)/np.std(b50[halfinds],axis=1,ddof=1)
    ci_s=np.nanquantile(corrs,[.025,.975]).tolist();ci_diff=np.nanquantile(differences,[.025,.975]).tolist()
    spcorr=float(spearmanr(a,b).statistic)
    h1=ratio>=3;h2=spcorr>=.5 and ci_s[0]>0;h3=dif>=.4 and ci_diff[0]>0
    h4=summary['functions']['mix_100']['classic']['bic_1_minus_bic_2']>10 and summary['functions']['mix_50']['classic']['bic_1_minus_bic_2']<=5
    summary['hypothesis_tests']={'H1_sensitivity':bool(h1),'H2_matched_correlation':bool(h2),'H3_old_SP_group_separation':bool(h3),
          'H4_bimodality':bool(h4),'primary_gate_H1_H2_H3':bool(h1 and h2 and h3),
          'classic_sd_ratio_100_over_50':ratio,
          'classic_sd_ratio_bootstrap95':np.quantile(sdboot,[.025,.975]).tolist(),
          'matched_spearman':spcorr,'matched_spearman_bootstrap95':ci_s,
          'old_SP_high_group_n':int(sum(labels)),'old_SP_low_group_n':int(sum(~labels)),
          'classic_cost_high_minus_low':dif,'classic_cost_group_difference_bootstrap95':ci_diff}
    (ROOT/'results.json').write_text(json.dumps({'summary':summary,'runs':rows},ensure_ascii=False,indent=2))
    print('RESULTS',json.dumps(summary['hypothesis_tests'],indent=2),flush=True)
    print('BIC',[(m,summary['functions'][m]['classic']['bic_1_minus_bic_2']) for m in ('mix_50','mix_100')],flush=True)
    print('SAVED',ROOT/'results.json',flush=True)

if __name__=='__main__':
    if sys.argv[1]=='chunk':go_chunk(int(sys.argv[2]),int(sys.argv[3]))
    elif sys.argv[1]=='finish':finish()
    else: raise ValueError('command')
