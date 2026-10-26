#!/usr/bin/env python3
"""Random exhaustive verification of the word-boundary DAG extrema."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from boundary_lattice_certify import score_bounds_dp,naive_relaxed_bounds,brute_force

def main():
    rng=np.random.default_rng(20261023)
    checked=0
    strict=0
    for n in range(2,7):
        for t in range(125):
            p=1
            orig=rng.integers(-5,6,size=(n,p),dtype=np.int32)
            amin=rng.integers(-8,5,size=(n,p),dtype=np.int32)
            amax=amin+rng.integers(0,12,size=(n,p),dtype=np.int32)
            joins=rng.integers(-8,9,size=(n-1,p),dtype=np.int32)
            kmax=min(3,n)
            dl,dh=score_bounds_dp(orig,amin,amax,joins,kmax)
            nl,nh=naive_relaxed_bounds(orig,amin,amax,joins,kmax)
            for k in range(kmax+1):
                low,high=brute_force(orig[:,0],amin[:,0],amax[:,0],joins[:,0],k)
                assert dl[k,0]==low and dh[k,0]==high, (n,t,k,dl[k,0],dh[k,0],low,high)
                assert nl[k,0]<=dl[k,0] and dh[k,0]<=nh[k,0]
                if k==1:assert nl[k,0]==dl[k,0] and nh[k,0]==dh[k,0]
                if nl[k,0]<dl[k,0] or dh[k,0]<nh[k,0]:strict+=1
                checked+=1
    print('PASS',checked,'exhaustive bounds,',strict,'strict improvements over overlap relaxation')

if __name__=='__main__':main()
