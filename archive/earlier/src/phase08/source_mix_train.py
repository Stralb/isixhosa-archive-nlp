#!/usr/bin/env python3
"""Frozen Phase8B source mix dose-response; uses exactly Phase8 word_uniform sampler."""
import hashlib,json,sys,time
from pathlib import Path
import run_phase8 as r
ROOT=r.ROOT
SHA='e977'  # overwritten with precommitted actual digest below
EXPECTED=Path(ROOT/'SOURCE_MIX_PRECOMMIT_SHA256.txt').read_text().split()[0]
assert hashlib.sha256((ROOT/'SOURCE_MIX_PRECOMMIT.md').read_bytes()).hexdigest()==EXPECTED
NEW_MIXES=(.75,.9,.95)
SEEDS=r.SEEDS
PARAMS=[(mix,seed) for mix in NEW_MIXES for seed in SEEDS]
CP=ROOT/'source_mix_checkpoint.json'

def main():
    start=int(sys.argv[1]);stop=int(sys.argv[2]);
    rows=json.loads(CP.read_text())['runs'] if CP.exists() else []
    assert start==len(rows)
    data=r.issue_data()
    t=time.monotonic()
    for i in range(start,stop):
        mix,seed=PARAMS[i]
        row=r.train_one(data,'word_uniform',mix,seed)
        rows.append(row)
        CP.write_text(json.dumps({'precommit_sha256':EXPECTED,'runs':rows},indent=2))
        print('TRAINED MIX',len(rows),'/',len(PARAMS),row['name'],'elapsed',round(time.monotonic()-t,2),flush=True)
    print('CHUNK FINISHED',len(rows))
if __name__=='__main__':main()
