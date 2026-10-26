#!/usr/bin/env python3
"""Prospectively frozen, post-main-result line-preserving BPE placebo controls."""
from pathlib import Path
import csv,json,random,collections
import numpy as np
import sentencepiece as spm
D=Path(__file__).resolve().parent;P=D/'inputs'
W=[json.loads(x) for x in (P/'windows.jsonl').read_text().splitlines()];W=[w for w in W if w['year']==1888]
assert len(W)==207
SOURCES=('IMVO','ISIG');SEEDS=range(3)
def nch(line):return sum(c.isalpha() for c in line)
def half_split(lines,seed,target=30000):
 n=len(lines); lens=[nch(t) for t in lines]
 assert sum(lens)==2*target
 order=list(range(n));random.Random(seed).shuffle(order)
 mask=(1<<(target+1))-1
 steps=[1]
 for i in order:steps.append((steps[-1]|(steps[-1]<<lens[i]))&mask)
 assert steps[-1] & (1<<target)
 rem=target;chosen=set()
 for j in range(n,0,-1):
  prev=steps[j-1]
  if prev & (1<<rem):continue
  i=order[j-1];assert lens[i]<=rem
  chosen.add(i);rem-=lens[i]
 assert rem==0
 A=[line for i,line in enumerate(lines) if i in chosen];B=[line for i,line in enumerate(lines) if i not in chosen]
 assert sum(map(nch,A))==target and sum(map(nch,B))==target
 assert len(A)+len(B)==n
 return A,B

models={};audit=[]
for k in SEEDS:
 groups={'A':[],'B':[]}
 for s in SOURCES:
  lines=(P/f'train_{s}_seed{k}.txt').read_text().splitlines()
  a,b=half_split(lines,30524+100*k+(0 if s=='IMVO' else 1))
  groups['A']+=a;groups['B']+=b
  for v,part in (('A',a),('B',b)):
   audit.append(dict(seed=k,source=s,mixed_label=v,source_chars=sum(map(nch,part)),source_lines=len(part),full_lines_preserved=1))
 for v in ('A','B'):
  seq=groups[v];random.Random(30924+k+(v=='B')).shuffle(seq)
  assert sum(map(nch,seq))==60000
  path=D/f'lineplacebo_{v}_seed{k}.txt';path.write_text('\n'.join(seq)+'\n',encoding='utf8')
  prefix=str(D/f'LINE_PLACEBO_{v}_seed{k}')
  spm.SentencePieceTrainer.train(input=str(path),model_prefix=prefix,model_type='bpe',vocab_size=1000,character_coverage=1.0,normalization_rule_name='identity',hard_vocab_limit=False,unk_id=0,bos_id=-1,eos_id=-1,pad_id=-1,shuffle_input_sentence=False,minloglevel=2)
  models[v,k]=spm.SentencePieceProcessor(model_file=prefix+'.model')
assert len(models)==6
with (D/'line_placebo_training_audit.csv').open('w',newline='') as f:
 wr=csv.DictWriter(f,fieldnames=list(audit[0]));wr.writeheader();wr.writerows(audit)

cache={(v,k):{} for v in ('A','B') for k in SEEDS}
def npiece(v,k,t):
 c=cache[v,k]
 if t not in c:c[t]=len(models[v,k].encode(t))
 return c[t]
rows=[]
for w in W:
 for k in SEEDS:
  length=sum(nch(t) for t in w['tokens']);pa=sum(npiece('A',k,t) for t in w['tokens']);pb=sum(npiece('B',k,t) for t in w['tokens'])
  saved=(pb-pa) * (1 if w['source']=='IMVO' else -1)
  rows.append(dict(file=w['file'],issue=w['issue'],source=w['source'],seed=k,chars=length,A_pieces=pa,B_pieces=pb,oriented_saved_pieces=saved))
with (D/'line_placebo_page_scores.csv').open('w',newline='') as f:
 wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
fixed=list(csv.DictReader((D/'source_page_shared_decomposition.csv').open()))
def to_cells(rows,charcol,savecol):
 cs=collections.defaultdict(lambda:np.zeros(2,dtype=float))
 for r in rows:cs[(r['source'],int(r['seed']),r['issue'])]+=np.array([float(r[charcol]),float(r[savecol])])
 return cs
null=to_cells(rows,'chars','oriented_saved_pieces');actual=to_cells(fixed,'all_alphabetic_chars','all_specialist_saved_pieces')
issues={s:sorted({z['issue'] for z in rows if z['source']==s}) for s in SOURCES}
def sc(cells,s,k,ids):
 v=sum((cells[(s,k,i)] for i in ids),np.zeros(2));return 100*v[1]/v[0]
per_seed=[]
for k in SEEDS:
 v=sum(sc(actual,s,k,issues[s]) for s in SOURCES);q=sum(sc(null,s,k,issues[s]) for s in SOURCES)
 per_seed.append(dict(seed=k,actual_sum=v,line_preserving_placebo_sum=q,paired_difference=v-q))

rng=np.random.default_rng(30924);boot=[]
for _ in range(2000):
 k=int(rng.integers(0,3));diff=0
 for s in SOURCES:
  idx=rng.choice(issues[s],size=len(issues[s]),replace=True)
  diff+=sc(actual,s,k,idx)-sc(null,s,k,idx)
 boot.append(diff)
ci=list(map(float,np.quantile(boot,[.025,.975])))
mean_actual=float(np.mean([r['actual_sum'] for r in per_seed])); mean_null=float(np.mean([abs(r['line_preserving_placebo_sum']) for r in per_seed]))
out={'classification':'post-result prospectively frozen sensitivity, not independent confirmation','per_seed':per_seed,'mean_actual_S':mean_actual,'mean_abs_line_placebo_S':mean_null,'paired_issue_seed_bootstrap_delta_ci95':ci,
     'passes_sensitivity_gate': bool(mean_actual>2*mean_null and ci[0]>0),
     'train_lines_preserved':True,'training_char_budget':60000,'source_mix_per_control':{'IMVO':30000,'ISIG':30000},'test_windows':len(W),'test_issues':{s:len(issues[s]) for s in SOURCES}}
(D/'LINE_PLACEBO_RESULTS.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
