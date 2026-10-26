from pathlib import Path
import pickle,json,collections,regex,numpy as np
O=Path('/mnt/data/isiXhosa_three_track_pilot')
P=pickle.load(open(O/'corpus_cache.pkl','rb')); D={r['id']:r for r in P}
Q=[json.loads(s) for s in open(O/'query_repaired.jsonl')]
R={(r['method'],r['kind'],r['qid']):r for r in (json.loads(s) for s in open(O/'retrieval_repaired_results.jsonl'))}
pat=regex.compile(r'(?V1)\p{L}+(?:[\'’]\p{L}+)*')
def words(s):return [w.lower() for w in pat.findall(s)]
source=dict()
for q in Q:
 w=words(D[q['page']]['raw']);needle=words(q['query'])
 pos=next(i for i in range(len(w)-9) if w[i:i+10]==needle)
 q['relative_position']=pos/max(len(w)-10,1)
 q['source_page_num']=int(q['page'].split('_p')[-1].split('.')[0])
 q['match_other_newspaper']=any(d.split('_')[0]!=q['source'] for d in q['match_pages'])
 q['matches_ge10']=q['num_exact_match_pages']>=10

def stats(x):
 if not x:return None
 return {'n':len(x),'multi_frac':round(float(np.mean([q['num_exact_match_pages']>1 for q in x])),4),
   'high_repeat_ge10':round(float(np.mean([q['matches_ge10'] for q in x])),4),
   'cross_newspaper_frac':round(float(np.mean([q['match_other_newspaper'] for q in x])),4),
   'raw_exact_top1':round(float(np.mean([R[('raw','clean',q['id'])]['exact_r1'] for q in x])),4),
   'raw_any_top1':round(float(np.mean([R[('raw','clean',q['id'])]['any_r1'] for q in x])),4)}
res={}
for src in ('IMVO','ISIG'):
 for lan in ('xh','en'):
  group=[q for q in Q if q['source']==src and q['lang']==lan]
  for label,ss in [('all',group),('p001',[q for q in group if q['source_page_num']==1]),
                    ('p002_p003',[q for q in group if 2<=q['source_page_num']<=3]),
                    ('p004_plus',[q for q in group if q['source_page_num']>=4]),
                    ('interior_position_10_90',[q for q in group if .1<=q['relative_position']<=.9]),
                    ('interior_non_frontpage',[q for q in group if .1<=q['relative_position']<=.9 and q['source_page_num']>1]),
                    ('excluding_ge10_repeated',[q for q in group if not q['matches_ge10']])]:
   k=src+'_'+lan+'_'+label;res[k]=stats(ss)
   print(k,res[k])
res['all']=stats(Q)
res['nonfront']=stats([q for q in Q if q['source_page_num']>1]);res['nonfront_interior']=stats([q for q in Q if q['source_page_num']>1 and .1<=q['relative_position']<=.9])
(O/'boilerplate_control_summary.json').write_text(json.dumps(res,indent=2))
with (O/'boilerplate_query_metadata.jsonl').open('w') as f:
 for q in Q:
  z={k:q[k] for k in ('id','source','lang','source_page_num','relative_position','num_exact_match_pages','num_exact_match_issues','match_other_newspaper','matches_ge10')}
  f.write(json.dumps(z)+'\n')
print('all',res['all'],'nonfront',res['nonfront'],'nonfront_interior',res['nonfront_interior'])
