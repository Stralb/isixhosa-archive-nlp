#!/usr/bin/env python3
"""Audit Phase2's synthetic same-page shortcut; diagnostic NOT a deployable model."""
import argparse,csv
from pathlib import Path
import numpy as np

def run(path,out):
 d=np.load(path,allow_pickle=False)
 masks=d['masks'];n=d['page_totals'];A=np.minimum(d['base_counts']/n[:,None],.02)
 B=np.minimum(d['injected_counts']/n[:,None],.02)
 truth=d['truth'][1]
 changed=np.where(np.abs(truth)>1e-14)[0]
 unchanged=np.where(np.abs(truth)<=1e-14)[0]
 assert len(changed)==12 and len(unchanged)==68
 rows=[]
 for si,scenario in enumerate(d['scenarios']):
  correct=0;bad=0;total=0;null_calls=0;intersections=[]
  for rep in range(masks.shape[1]):
   jointly=masks[si,rep,0]&masks[si,rep,1]
   intersections.append(int(jointly.sum()))
   if not jointly.any():continue
   paired=(B[jointly]-A[jointly]).mean(axis=0)
   correct+=int((np.sign(paired[changed])==np.sign(truth[changed])).sum())
   bad+=int((np.sign(paired[changed])!=np.sign(truth[changed])).sum())
   null_calls+=int(np.sum(paired[unchanged]!=0))
   total+=len(changed)
  rows.append({'scenario':scenario,'replicates':masks.shape[1],
               'median_same_page_intersection':float(np.median(intersections)),
               'correct_signs':correct,'injected_sign_checks':total,
               'correct_sign_fraction':correct/total,'incorrect_signs':bad,
               'null_nonzero_page_deltas':null_calls,'null_checks':len(unchanged)*masks.shape[1]})
 Path(out).parent.mkdir(parents=True,exist_ok=True)
 with open(out,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 for row in rows:print(row['scenario'],row['correct_sign_fraction'],row['null_nonzero_page_deltas'])

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();run(a.input,a.out)
