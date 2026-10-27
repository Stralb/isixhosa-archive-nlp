#!/usr/bin/env python3
"""Check clean replay's strict equality with the Phase4 frozen numerical outputs."""
import argparse,hashlib,json
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(folder,replay):
 a=Path(folder);b=Path(replay);same=[];diff=[]
 for part in ('dev','test'):
  for name in ('audit_replicate_results.csv','audit_orders_hashes.csv','bounds_and_audits.npz',
               'scenario_summary.csv','pooled_summary.csv','RUN_MANIFEST.json'):
   aa=a/'outputs'/part/name;bb=b/part/name
   (same if sha(aa)==sha(bb) else diff).append(f'{part}/{name}')
 for f in ('phase4_audit_budget_width.png','phase4_audit_budget_sign_calls.png','phase4_zero_cell_support.png'):
  aa=a/'figures'/f;bb=b/'figures'/f
  (same if sha(aa)==sha(bb) else diff).append(f)
 out={'matched_byte_for_byte':same,'mismatched':diff,'matched_count':len(same),'pass':len(diff)==0}
 target=b/'verification'/'REPLAY_COMPARISON.json';target.write_text(json.dumps(out,indent=2)+'\n')
 print('Fresh-replay byte-level matches',len(same),'failures',len(diff))
 assert not diff,diff

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase4',required=True);p.add_argument('--replay',required=True);s=p.parse_args();main(s.phase4,s.replay)
