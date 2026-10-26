#!/usr/bin/env python3
import collections,hashlib,json,random,statistics,itertools,math,sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon
import run_phase9 as ph
p8=ph.p8; ROOT=ph.ROOT
assert len(json.loads((ROOT/'training_checkpoint.json').read_text())['runs'])==40
PAGES=[p for p in p8.training_pages_cached if p['source']=='IMVO']
words=[];issues=[];months=[]
for p in PAGES:
    ww=p8.forms(p['text']);words.extend(ww)
    issues.extend([f'{p["year"]}-{p["month"]}-{p["day"]}']*len(ww))
    months.extend([int(p['month'])]*len(ww))
assert words==ph.allwords['IMVO']
full=collections.Counter(words)
full_types=set(full)
full_top=set(x for x,_ in full.most_common(100))

def indexed_sample(seed):
    idx=list(range(len(words)))
    random.Random(seed*101+1).shuffle(idx)
    out=[];remaining=p8.BUDGET
    for i in idx:
        n=p8.alpha_len(words[i])
        if 0<n<=remaining:
            out.append(i);remaining-=n
        if not remaining: break
    assert remaining==0
    sampled=[words[i] for i in out]
    assert collections.Counter(sampled)==collections.Counter(ph.get_sample(1.,seed))
    return out

res=json.loads((ROOT/'results.json').read_text()); by={r['seed']:r for r in res['runs'] if r['mix']==1.}
r8=json.loads(p8.OUT.read_text())
prior={r['seed']:r['validation']['worst_unseen_E'] for r in r8['runs'] if r['method']=='word_uniform' and r['mix']==1.}
features=[]
for seed in ph.SEEDS:
    ix=indexed_sample(seed)
    types=collections.Counter(words[i] for i in ix)
    n=len(ix);ct=collections.Counter(issues[i] for i in ix)
    icounts=np.array(list(ct.values()),dtype=np.float64)
    freq=np.array(list(types.values()),dtype=np.float64)
    probs=freq/freq.sum()
    ms=collections.Counter(months[i] for i in ix)
    concat_alphabets={c for w in types for c in w}
    row={'seed':seed,'group':('higher_cost' if prior[seed]>=39.5 else 'lower_cost'),
      'worst_unseen_E':by[seed]['validation']['worst_unseen_E'],
      'self_train_E':by[seed]['self_train_E'],
      'type_count':len(types),
      'word_entropy_bits':float(-sum(probs*np.log2(probs))),
      'top100_full_pool_share':sum(n for w,n in types.items() if w in full_top)/n,
      'full_pool_lowfreq_2_share':sum(v for w,v in types.items() if full[w]<=2)/n,
      'full_pool_lowfreq_5_share':sum(v for w,v in types.items() if full[w]<=5)/n,
      'singleton_in_sample_share':sum(v for w,v in types.items() if v==1)/n,
      'mean_word_alpha':p8.BUDGET/n,
      'hyphen_apostrophe_word_share':sum(v for w,v in types.items() if '-' in w or "'" in w or '’' in w)/n,
      'alphabet_size':len(concat_alphabets),
      'issue_max_share':float(icounts.max()/n),
      'issue_top5_share':float(sum(sorted(icounts,reverse=True)[:5])/n),
      'issue_hhi':float(sum((icounts/n)**2)),
      'month_shares':{str(m):round(v/n,5) for m,v in sorted(ms.items())}}
    features.append(row)

labels=[row['group'] for row in features]
group_means={}
for field in ['type_count','word_entropy_bits','top100_full_pool_share','full_pool_lowfreq_2_share','full_pool_lowfreq_5_share','singleton_in_sample_share','mean_word_alpha','hyphen_apostrophe_word_share','alphabet_size','issue_max_share','issue_top5_share','issue_hhi','self_train_E']:
    high=np.array([r[field] for r in features if r['group']=='higher_cost']);low=np.array([r[field] for r in features if r['group']=='lower_cost'])
    values=np.array([r[field] for r in features]);performance=np.array([r['worst_unseen_E'] for r in features])
    group_means[field]={'higher_cost_group':float(high.mean()),'lower_cost_group':float(low.mean()),
           'higher_minus_lower':float(high.mean()-low.mean()),
           'spearman_with_cost':float(spearmanr(values,performance).statistic)}

models={}
for seed in ph.SEEDS:
    j=json.loads((ph.MODELS/f'classic_mix100_draw{seed}.json').read_text())
    # total vocabulary set derived from actual learned merges, including characters
    models[seed]={'vocab':set(j['alphabet'])|{''.join(p) for p in j['merges']},
                  'pairs':set(tuple(p) for p in j['merges']),
                  'first_merge':j['merges'][:20],
                  'alphabet':set(j['alphabet'])}

jac={'lower_lower':[],'higher_higher':[],'cross_group':[]}
for s1,s2 in itertools.combinations(ph.SEEDS,2):
    s1l=prior[s1]>=39.5;s2l=prior[s2]>=39.5
    same= 'higher_higher' if s1l and s2l else 'lower_lower' if not s1l and not s2l else 'cross_group'
    a=models[s1]['vocab'];b=models[s2]['vocab'];jac[same].append(len(a&b)/len(a|b))

highset=[models[s]['vocab'] for s in ph.SEEDS if prior[s]>=39.5]
lowset=[models[s]['vocab'] for s in ph.SEEDS if prior[s]<39.5]
allkeys=set().union(*highset,*lowset)
# Deltas in vocab *membership*; not causal factors.
exclusive=[]
for key in allkeys:
    high_rate=sum(key in a for a in highset)/len(highset)
    low_rate=sum(key in a for a in lowset)/len(lowset)
    if abs(high_rate-low_rate)>=.75:
        exclusive.append({'symbol':key,'high_rate':high_rate,'low_rate':low_rate,'difference':high_rate-low_rate})
exclusive=sorted(exclusive,key=lambda x:(-abs(x['difference']),-len(x['symbol']),x['symbol']))

out={'status':'exploratory - group labels from previously inspected validation',
    'sample_index_reconstruction_counter_verified':True,
    'features':features,'feature_group_comparison':group_means,
    'vocab_jaccard':{k:{'mean':statistics.mean(v),'std':statistics.stdev(v),'n_pairs':len(v)} for k,v in jac.items()},
    'features_exclusive_vocab_symbols':exclusive[:120],
    'distinct_exclusive_symbols_count':len(exclusive),
    'first20_merges_low_sample':models[1001]['first_merge'],
    'first20_merges_high_sample':models[1006]['first_merge']}
(ROOT/'mechanism_audit_results.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
print('GROUP COMPARISONS')
for name,x in sorted(group_means.items(),key=lambda z:-abs(z[1]['spearman_with_cost'])):
    print(name,'hi',round(x['higher_cost_group'],4),'lo',round(x['lower_cost_group'],4),'diff',round(x['higher_minus_lower'],4),'rho',round(x['spearman_with_cost'],3))
print('VOCAB JACCARD',out['vocab_jaccard'])
print('SELECTIVE SYMBOLS COUNT',len(exclusive),'TOP 20',exclusive[:20])
