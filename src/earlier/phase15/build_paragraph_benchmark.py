"""Deterministic, paragraph-faithful benchmark/audit. See PRECOMMIT.md."""
from pathlib import Path
import pickle,json,hashlib,re,regex,collections,time
P=Path('/mnt/data/isiXhosa_three_track_pilot'); O=Path('/mnt/data/isiXhosa_phase15_source_grounded_search');O.mkdir(exist_ok=True)
start=time.time();doclist=pickle.load(open(P/'corpus_cache.pkl','rb'));orig=[json.loads(s) for s in open(P/'query_repaired.jsonl')]
origbyid={q['page']:[] for q in orig}
for q in orig:origbyid[q['page']].append(q)
W=regex.compile(r"(?V1)\p{L}+(?:['’]\p{L}+)*")
def words(s):return tuple(w.lower() for w in W.findall(s))
def paragraphs(s):return [words(x) for x in re.split(r'\n\s*\n',s) if x.strip()]
def gramset(par,k=10):return {p[i:i+k] for p in par for i in range(len(p)-k+1)}
def valid(g):return len(set(g))>=5 and sum(len(w) for w in g)>42
stats=collections.defaultdict(lambda:collections.Counter()); audits=[];qs=[];old=[]
for di,d in enumerate(doclist):
 rawparas=paragraphs(d['raw']);rp=gramset(rawparas)
 flat_raw=words(d['raw']); flat_raw_set={flat_raw[i:i+10] for i in range(len(flat_raw)-9)}
 rec={'id':d['id'],'source':d['source'],'issue':d['issue'],'page_num':int(d['page'][1:]),'raw_paragraphs':len(rawparas)}
 for lang in ['xh','en']:
  vparas=paragraphs(d[lang]); vflat=words(d[lang]); rkey=d['source']+'_'+lang
  c=stats[rkey];c['pages']+=1
  view_in_par=set(); n_windows=0; in_rawpara=0; in_rawflat=0; in_viewpara=0
  for v in vparas:
   for i in range(len(v)-9):view_in_par.add(v[i:i+10])
  for i in range(len(vflat)-9):
   g=vflat[i:i+10];n_windows+=1
   in_viewpara+=(g in view_in_par)
   in_rawpara+=(g in rp)
   in_rawflat+=(g in flat_raw_set)
  c['view_windows']+=n_windows;c['in_view_para']+=in_viewpara;c['in_raw_para']+=in_rawpara;c['in_raw_flat']+=in_rawflat
  rec[lang+'_window_count']=n_windows;rec[lang+'_not_raw_flat']=n_windows-in_rawflat;rec[lang+'_not_raw_para']=n_windows-in_rawpara;rec[lang+'_not_view_para']=n_windows-in_viewpara
  if d['id'] in origbyid:
   for q in origbyid[d['id']]:
    if q['lang']!=lang:continue
    g=words(q['query']); assert g in flat_raw_set,('gold not in flat raw',q['id'])
    old.append({'id':q['id'],'page':q['page'],'source':q['source'],'lang':lang,'issue':q['issue'],
     'page_num':int(d['page'][1:]),'num_exact_match_pages_flat':q['num_exact_match_pages'],
     'in_raw_paragraph':g in rp,'in_view_paragraph':g in view_in_par,
     'originally_repaired':bool(q['repaired'])})
    # Choose paragraph-faithful query with minimal seeded hash among all eligible candidates.
    best=None; besthash=None; count=0
    for pidx,v in enumerate(vparas):
     for wi in range(len(v)-9):
      x=v[wi:wi+10]
      if x not in rp or not valid(x):continue
      count+=1
      h=hashlib.blake2b((q['id']+'#phase15#'+str(pidx)+'#'+str(wi)).encode('utf8'),digest_size=8).digest()
      if besthash is None or h<besthash:besthash=h;best=x
    if best is not None:
     assert best in rp and len(best)==10
     flat_pos=next((i for i in range(len(flat_raw)-9) if flat_raw[i:i+10]==best),None)
     assert flat_pos is not None
     qs.append(dict(id=q['id'],page=q['page'],issue=q['issue'],source=q['source'],lang=q['lang'],
        query=' '.join(best),relative_position=flat_pos/max(len(flat_raw)-10,1),
        source_page_num=int(d['page'][1:]),eligible_windows=count,
        old_query_para_valid=bool(g in rp)))
    else:rec[f'no_{lang}_valid_candidate']=True
 audits.append(rec)
 if (di+1)%600==0:print('audit pages',di+1,'elapsed',round(time.time()-start,1),flush=True)
assert len(old)==len(orig)
print('SELECTED',len(qs),'of',len(orig),flush=True)
# Confirm multiple paragraph-verbatim source pages for new queries, and old query alternative golds.
queryset=collections.defaultdict(list)
for q in qs:queryset[tuple(q['query'].split())].append(('new',q['id']))
for q in orig:queryset[words(q['query'])].append(('old',q['id']))
found=collections.defaultdict(set)
for j,d in enumerate(doclist):
 for g in gramset(paragraphs(d['raw'])):
  if g in queryset:
   for key in queryset[g]:found[key].add(d['id'])
 if (j+1)%1000==0:print('gold matches',j+1,'elapsed',round(time.time()-start,1),flush=True)
oldmap={x['id']:x for x in old};new_count=collections.Counter();old_count=collections.Counter()
for q in qs:
 hits=sorted(found[('new',q['id'])]);assert q['page'] in hits
 q['match_pages']=hits;q['num_exact_match_pages']=len(hits);q['num_exact_match_issues']=len({x.rsplit('_',1)[0] for x in hits})
 new_count[q['source']+'_'+q['lang']]+=1
for q in orig:
 d=oldmap[q['id']];hits=sorted(found[('old',q['id'])]);d['any_raw_paragraph_page_count']=len(hits)
 d['has_any_raw_paragraph_page']=len(hits)>0
 if d['in_raw_paragraph']:assert q['page'] in hits
 old_count[d['source']+'_'+d['lang']]+=int(d['in_raw_paragraph'])
# Fixed deterministic synthetic noise cloned from prior benchmark; source qid -> same perturbation schedule, new original text.
import random
for q in qs:
 toks=q['query'].split();r=random.Random(int(hashlib.sha256((q['id']+'#noise20261023').encode()).hexdigest()[:16],16));elig=[i for i,w in enumerate(toks) if len(w)>=4];r.shuffle(elig)
 for j in elig[:max(1,round(len(toks)*.25))]:
  w=toks[j];pos=r.randrange(1,len(w)-1);c=r.choice([v for v in 'aeinorsutkgmhbd' if v!=w[pos]]);toks[j]=w[:pos]+c+w[pos+1:]
 k=r.randrange(1,len(toks)-1)
 if r.randrange(2)==0 and len(toks[k])>5:
  pos=r.randrange(2,len(toks[k])-1);toks[k:k+1]=[toks[k][:pos],toks[k][pos:]]
 else:toks[k:k+2]=[toks[k]+toks[k+1]]
 q['noisy_query']=' '.join(toks)
for fn,arr in [('page_audit.jsonl',audits),('original_benchmark_paragraph_audit.jsonl',old),('paragraph_queries.jsonl',qs)]:
 with (O/fn).open('w') as f:
  for x in arr:f.write(json.dumps(x,ensure_ascii=False)+'\n')
S={k:{'pages':int(v['pages']),'view_windows':int(v['view_windows']),
'not_verbatim_flat':int(v['view_windows']-v['in_raw_flat']),
'not_verbatim_paragraph':int(v['view_windows']-v['in_raw_para']),
'view_stitched_windows':int(v['view_windows']-v['in_view_para']),
'not_verbatim_flat_frac':round(1-v['in_raw_flat']/v['view_windows'],6) if v['view_windows'] else None,
'not_verbatim_paragraph_frac':round(1-v['in_raw_para']/v['view_windows'],6) if v['view_windows'] else None}
for k,v in stats.items()}
S['old_query_audit']={'total':len(old),'in_raw_paragraph':sum(o['in_raw_paragraph'] for o in old),'in_view_paragraph':sum(o['in_view_paragraph'] for o in old),
  'by_source_language':{k:{'total':sum(x['source']+'_'+x['lang']==k for x in old),'raw_paragraph_valid':v,'view_paragraph_valid':sum(x['in_view_paragraph'] for x in old if x['source']+'_'+x['lang']==k)} for k,v in old_count.items()}}
S['new_query_counts']=dict(new_count)
S['new_query_total']=len(qs)
S['new_query_repeated_n']=sum(q['num_exact_match_pages']>1 for q in qs)
S['new_query_nonfront_n']=sum(q['source_page_num']>1 for q in qs)
S['elapsed_s']=round(time.time()-start,2)
(O/'paragraph_benchmark_summary.json').write_text(json.dumps(S,indent=2))
print(json.dumps(S,indent=2),flush=True)
