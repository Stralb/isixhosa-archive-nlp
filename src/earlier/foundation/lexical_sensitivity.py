"""Retrospective, post-audit sensitivity; no independent year content is touched."""
import collections, json, math
from pathlib import Path
from audit_overlap import load_pages,index_ngrams,digest
ROOT=Path('/mnt/data/isiXhosa_publication_continuation')
N=12
pages=load_pages(); train=[p for p in pages if p['split']=='train']; oldtest=[p for p in pages if p['split']=='old_test']
index,_=index_ngrams(train,N)
raw=collections.defaultdict(collections.Counter)
novel=collections.defaultdict(collections.Counter)
removed=collections.defaultdict(collections.Counter)
pagepos=collections.defaultdict(lambda: [0,0,0])
for p in oldtest:
 ts=p['tokens']; masks=bytearray(len(ts))
 for i in range(len(ts)-N+1):
  h=digest(ts[i:i+N]); m=index.get(h,())
  if any(d!=p['issue'] for d,s in m):masks[i:i+N]=b'\x01'*N
 raw[p['source']].update(ts)
 novel[p['source']].update(w for w,flag in zip(ts,masks) if not flag)
 removed[p['source']].update(w for w,flag in zip(ts,masks) if flag)
 doc_page=int(p['name'].split('_p')[-1].split('.')[0]); rec=pagepos[(p['source'],doc_page)]
 rec[0]+=len(ts);rec[1]+=sum(masks);rec[2]+=1

def jsd(a,b):
 va=sum(a.values()); vb=sum(b.values());
 out=0
 for word in a.keys()|b.keys():
  pa=a[word]/va; pb=b[word]/vb; avg=(pa+pb)/2
  if pa:out+=.5*pa*math.log2(pa/avg)
  if pb:out+=.5*pb*math.log2(pb/avg)
 return out

report={'old_test_year':1888,'ngram_n':N,'unigram_jsd_bits_raw':jsd(raw['IMVO'],raw['ISIG']), 'unigram_jsd_bits_without_1887_train_overlap':jsd(novel['IMVO'],novel['ISIG']),
'source':{},'page_positions':{str(k):{'tokens':x[0],'overlap_tokens':x[1], 'overlap_rate':x[1]/x[0] if x[0] else None,'pages':x[2]} for k,x in pagepos.items()}}
for s in ['IMVO','ISIG']:
 total=sum(raw[s].values()); rr=sum(removed[s].values())
 high=[]
 for w,rrr in removed[s].most_common():
  n=raw[s][w];
  if n>=10 and rrr>=5 and len(w)>3:
   high.append({'surface_form':w,'raw_n':n,'overlap_n':rrr,'fraction_token_count_in_matching_passages':rrr/n})
 high.sort(key=lambda x:(-x['overlap_n'],-x['fraction_token_count_in_matching_passages']))
 report['source'][s]={'total_tokens':total,'covered_tokens':rr,'covered_fraction':rr/total,'high_overlap_surface_forms':high[:30]}
(ROOT/'lexical_sensitivity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print('JS divergence bits, original vs removal:',round(report['unigram_jsd_bits_raw'],5),round(report['unigram_jsd_bits_without_1887_train_overlap'],5))
for k,x in sorted(pagepos.items()):print(k,round(x[1]/x[0]*100,2),x[0],x[2])
for s in ['IMVO','ISIG']:
 print(s,report['source'][s]['covered_fraction'],report['source'][s]['high_overlap_surface_forms'][:12])
