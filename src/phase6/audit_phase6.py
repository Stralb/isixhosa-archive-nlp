#!/usr/bin/env python3
"""Read-only final science audit. Never fits a model or inspects new research data.

Recalculates model/passage pairing, validation lock, all 32 phase-3 and phase-5
factorial point effects and the selected primary BPC from frozen per-example CSVs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ARCHIVE_HASH = '375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5'
PROTOCOL_HASH = 'bc443d883fa1c2b8eb09ced5f87d8f65b352d55512ef2c594b1b23e64d496bf4'
FAMS = ('bpe','unigram')
ARMS = ('PP','PW','WP','WW')
SEEDS = (441,442,443,444,445)
PERIODS = ('1888','1889-94')
VIEWS = ('en','xh')

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

def require(cond, note):
    if not bool(cond): raise AssertionError(note)

def ctr(e):
    return {
        'allocation': 0.5*((e['WP']-e['PP'])+(e['WW']-e['PW'])),
        'within_view':0.5*((e['PW']-e['PP'])+(e['WW']-e['WP'])),
        'interaction':(e['WW']-e['WP'])-(e['PW']-e['PP']),
        'WW_minus_PP':e['WW']-e['PP'],
    }

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--archive',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();r=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    checks=[]
    def ok(name,value):checks.append({'test':name,'status':'PASS','evidence':value})
    require(sha(a.archive)==ARCHIVE_HASH,'wrong original language archive');ok('source_zip_sha256', ARCHIVE_HASH)
    require(sha(r/'Phase1/FROZEN_PROTOCOL.md')==PROTOCOL_HASH,'protocol modified');ok('frozen_protocol_sha256',PROTOCOL_HASH)
    sel=json.loads((r/'Phase2/outputs/PHASE2_SELECTION_FREEZE.json').read_text())
    require(sel['selected_alpha']==100 and sel['models_count']==40,'selection lock invalid')
    require(sel['heldout_content_reads']==0 and sel['validation_windows_per_model']==344,'validation lock invalid')
    require(sel['phase1_protocol_sha256']==PROTOCOL_HASH and sel['archive_sha256']==ARCHIVE_HASH,'selection source hash invalid')
    require(sel['candidate_grid']==[1.0,10.0,100.0],'alpha grid changed')
    require(all(sel['alpha_criteria'][i]['equal_view_family_arm_seed_mean_bpc'] > sel['alpha_criteria'][i+1]['equal_view_family_arm_seed_mean_bpc'] for i in range(2)),'alpha metrics inconsistent')
    ok('validation_selection_lock',{'alpha':100,'selected_from':[1,10,100],'windows_per_model':344,'validation_rows':sel['validation_scored_rows']})

    mt=pd.read_csv(r/'Phase2/outputs/model_training_manifest.csv')
    require(len(mt)==40,'training models != 40')
    require(set(mt.family)==set(FAMS) and set(mt.arm)==set(ARMS) and set(mt.seed)==set(SEEDS),'model design incomplete')
    require(mt.groupby(['family','arm']).size().eq(5).all(),'unbalanced model design')
    require(mt.actual_alpha_chars.between(499970,500030).all(),'training budget violation')
    require(mt.model_vocab_size.eq(2000).all(),'vocabulary size mismatch')
    require(mt.LM_training_documents.eq(545).all(),'LM training documents altered')
    require(mt.validation_windows.eq(344).all(),'validation windows mismatch')
    require(mt.validation_unknown_tokens.eq(0).all(),'unknown validation tokens')
    mism=[]
    for row in mt.itertuples():
        folder=r/'Phase2/outputs/models';name=row.model_name
        for suf,col in (('.model','model_sha256'),('.vocab','vocab_sha256')):
            f=folder/(name+suf)
            if not f.is_file() or sha(f)!=getattr(row,col):mism.append(str(f))
    require(not mism,f'{len(mism)} model hashes differ')
    exposure=(mt.groupby(['family','arm']).en_alpha_share.agg(['mean','min','max']).reset_index()).to_dict('records')
    ok('model_fits_and_hashes',{'trained_models':40,'training_budget_tolerance':30,'exposure':exposure})

    keys=['family','arm','seed','filename','view','issue','year','period']
    st=pd.read_csv(r/'Phase3/outputs/heldout_scores.csv')
    lm=pd.read_csv(r/'Phase5/outputs/heldout_lm_scores.csv')
    require(len(st)==len(lm)==99720,'row-count error')
    require(st[keys].duplicated().sum()==0 and lm[keys].duplicated().sum()==0,'duplicate score keys')
    pair=st.merge(lm,on=keys,suffixes=('_p3','_p5'),how='outer',indicator=True,validate='one_to_one')
    require((pair['_merge']=='both').all(),'Phase 3/5 score support mismatch')
    for col in ['words','chars','text_sha256','pieces','unknown']:
        require((pair[f'{col}_p3']==pair[f'{col}_p5']).all(),f'Phase 3/5 {col} mismatch')
    require(lm.unknown.eq(0).all(),'unexpected unknown tokens')
    require(np.allclose(lm.bpc, lm.bits/lm.chars,atol=1e-12,rtol=0),'bpc formula mismatch')
    docs=lm[['filename','view','issue','year','period','chars','words','text_sha256']].drop_duplicates()
    require(len(docs)==2493,'pageviews mismatch')
    require(docs.groupby('period').issue.nunique().to_dict()=={'1888':52,'1889-94':306},'issues mismatch')
    require(lm.groupby(['filename','view']).size().eq(40).all(),'unpaired heldout examples')
    require(lm.groupby(['family','arm','seed']).size().eq(2493).all(),'missing model prediction')
    ok('paired_evaluation_and_raw_score_consistency',{'phase3_rows':len(st),'phase5_rows':len(lm),'unique_pageviews':len(docs),'issues_1888':52,'issues_later':306,'unknown_tokens':0})

    phase3=pd.read_csv(r/'Phase3/outputs/factorial_contrasts.csv')
    phase5=pd.read_csv(r/'Phase5/outputs/bpc_factorial_contrasts.csv')
    require(len(phase3)==len(phase5)==32,'factorial count mismatch')
    calculated=[]; maximum={'phase3':0.,'phase5':0.}
    for family in FAMS:
      for period in PERIODS:
       for view in VIEWS:
        g=lm[(lm.family==family)&(lm.period==period)&(lm.view==view)]
        h=st[(st.family==family)&(st.period==period)&(st.view==view)]
        ds=g.groupby(['arm','seed'])[['bits','chars','pieces']].sum()
        dh=h.groupby(['arm','seed'])[['pieces','chars']].sum()
        require(np.array_equal(ds['pieces'].values,dh['pieces'].values) and np.array_equal(ds['chars'].values,dh['chars'].values),'aggregated token mismatch')
        bc={arm:np.asarray([ds.loc[(arm,seed),'bits']/ds.loc[(arm,seed),'chars'] for seed in SEEDS]) for arm in ARMS}
        pp={arm:np.asarray([100.*dh.loc[(arm,seed),'pieces']/dh.loc[(arm,seed),'chars'] for seed in SEEDS]) for arm in ARMS}
        for typ,data,reported in [('phase3',ctr(pp),phase3),('phase5',ctr(bc),phase5)]:
          for c,e in data.items():
            got=float(np.mean(e))
            existing=reported[(reported.family==family)&(reported.period==period)&(reported.view==view)&(reported.contrast==c)]
            require(len(existing)==1,'missing contrast '+str((typ,family,period,view,c)))
            delta=abs(got-float(existing.iloc[0]['mean']))
            require(delta<1e-10,'contrast mismatch '+str((typ,family,period,view,c,got,existing.iloc[0]['mean'])))
            maximum[typ]=max(maximum[typ],delta)
            if typ=='phase5':
                calculated.append({'family':family,'period':period,'view':view,'contrast':c,'value':got,'ci_low':float(existing.iloc[0].ci_low),'ci_high':float(existing.iloc[0].ci_high),'seed_directions': [int(np.sign(q)) for q in e]})
    ok('factorial_effects_recalculated',{'phase3_contrasts':32,'phase5_contrasts':32,'max_abs_residual':maximum})

    primary=next(x for x in calculated if x['family']=='bpe' and x['period']=='1888' and x['view']=='en' and x['contrast']=='allocation')
    later=next(x for x in calculated if x['family']=='bpe' and x['period']=='1889-94' and x['view']=='en' and x['contrast']=='allocation')
    f5=primary['value']<=-0.01 and primary['ci_high']<0 and later['value']<0
    require(f5,'F5 gate wrong')
    counter=next(x for x in calculated if x['family']=='bpe' and x['period']=='1889-94' and x['view']=='en' and x['contrast']=='WW_minus_PP')
    require(counter['value']>=0,'counterexample WW-PP unexpectedly changed')
    base=lm[(lm.family=='bpe')&(lm.period=='1888')&(lm.view=='en')&(lm.arm=='PP')].groupby('seed')[['bits','chars']].sum()
    base_bpc=(base.bits/base.chars).mean()
    ok('principal_claim_and_counterexample',{'F5_support':True,'primary_F5_1888':primary,'later_F5':later,'later_whole_policy_counterexample':counter,'PP_1888_English_reference_BPC':float(base_bpc),'primary_relative_magnitude_percent':float(-100*primary['value']/base_bpc)})
    # Important: five seeds only test technical stochasticity, 1888 previously inspected.
    warnings=[
      'The later-year corpus belongs to the same publication, not a new historical newspaper.',
      'The same held-out historical years were inspected in Phases 26-33; research-program external validation is NOT independent.',
      'Confidence intervals resample publication issues and hold model seeds fixed; no generalization to contemporary languages or people.',
      'EnTxt and XhTxt are archive block assignments, not human-verified gold-standard language labels.',
      'Exact-reprint filtering is a selective 13-word matching diagnostic, not proof of all-duplicate removal.',
      'Selected smoothing alpha=100 is the edge of prespecified {1,10,100} grid; do not claim exhaustive hyperparameter optimization.',
      'BPC measures a smoothed token bigram model, NOT neural accuracy, cultural understanding or societal harm.',
      'Source-data license and redistribution rights not established by these output files.',
    ]
    cert={'status':'SCIENTIFIC_AUDIT_PASS_SUBMISSION_NOT_YET_COMPLETE','tests':checks,'warnings':warnings,'phase5_key_effects':calculated,'source_required_external':str(Path(a.archive).name),'archive_sha256':ARCHIVE_HASH,'protocol_sha256':PROTOCOL_HASH}
    (out/'PHASE6_SCIENCE_AUDIT.json').write_text(json.dumps(cert,indent=2)+'\n',encoding='utf8')
    pd.DataFrame(calculated).drop(columns=['seed_directions']).to_csv(out/'PHASE6_VERIFIED_BPC_CONTRASTS.csv',index=False)
    print('PASS',len(checks),'independent scientific tests; point effects independently reproduced: 32 + 32')
    print('SOURCE',ARCHIVE_HASH,'MODELS',len(mt),'SCORE_ROWS',len(lm),'HELDOUT_PAGES',len(docs))
    print('F5',primary['value'],primary['ci_low'],primary['ci_high'],'later',later['value'])
    print('WW-PP LATER',counter['value'],'REFERENCE',float(base_bpc),'RELATIVE%',float(-100*primary['value']/base_bpc))

if __name__=='__main__':main()
