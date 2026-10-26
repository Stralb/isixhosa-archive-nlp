#!/usr/bin/env python3
"""Standalone, deterministic, frequency-weighted word-type BPE trainer.

No SentencePiece/tokenizers training code used. Boundaries differ by design:
each word is prefixed with a literal '▁' marker and words never cross boundaries.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import heapq
import json

class ClassicBPE:
    def __init__(self, alphabet=(), merges=()):
        self.alphabet=frozenset(alphabet)
        self.merges=[tuple(p) for p in merges]
        self.ranks={tuple(pair):i for i,pair in enumerate(self.merges)}

    @classmethod
    def fit(cls, words, vocab_size=2000, min_frequency=2, verbose=False):
        freq=Counter(words)
        alphabet=set(''.join(freq))
        alphabet.add('▁')
        symseq=[['▁',*w] for w in freq]
        weights=list(freq.values())
        links=defaultdict(set)
        count=Counter()
        for i,syms in enumerate(symseq):
            for pair,v in Counter(zip(syms,syms[1:])).items():
                links[pair].add(i)
                count[pair]+=v*weights[i]
        queue=[(-v,pair) for pair,v in count.items()]
        heapq.heapify(queue)
        inventory=set(alphabet)
        merges=[]
        while len(inventory)+1<vocab_size:
            while queue:
                neg,pair=heapq.heappop(queue)
                if count.get(pair,0)==-neg and count[pair]>=min_frequency:
                    break
            else:
                break
            if count[pair]<min_frequency:
                break
            hits=list(links[pair]); new_symbol=''.join(pair)
            changes=Counter()
            for i in hits:
                current=symseq[i]
                oldpairs=Counter(zip(current,current[1:]))
                result=[];j=0
                while j<len(current):
                    if j<len(current)-1 and (current[j],current[j+1])==pair:
                        result.append(new_symbol);j+=2
                    else:
                        result.append(current[j]);j+=1
                newpairs=Counter(zip(result,result[1:]))
                assert result!=current
                for q in oldpairs.keys()|newpairs.keys():
                    old=oldpairs.get(q,0);now=newpairs.get(q,0)
                    if old!=now:changes[q]+=(now-old)*weights[i]
                    if old and not now:
                        links[q].discard(i)
                    elif now and not old:
                        links[q].add(i)
                symseq[i]=result
            for q,dv in changes.items():
                new=count[q]+dv
                assert new>=0,('negative count',q,dv,count[q])
                if new==0:
                    count.pop(q,None)
                    if q in links:
                        assert len(links[q])==0,('stale index',q)
                        links.pop(q,None)
                else:
                    count[q]=new
                    heapq.heappush(queue,(-new,q))
            assert not links.get(pair,()),('merge should consume pair',pair)
            inventory.add(new_symbol)
            merges.append(pair)
            if len(queue)>4*len(count)+1000:
                queue=[(-v,p) for p,v in count.items() if v>=min_frequency]
                heapq.heapify(queue)
            if verbose and len(merges)%300==0:
                print('MERGE',len(merges),'VOCAB',len(inventory)+1,'TYPES',len(freq),flush=True)
        return cls(sorted(alphabet),merges)

    @lru_cache(maxsize=150000)
    def pieces(self,word):
        if any(c not in self.alphabet for c in word):
            return None
        syms=['▁',*word]
        while len(syms)>1:
            best_rank=None;best_pair=None
            for a,b in zip(syms,syms[1:]):
                p=(a,b); rank=self.ranks.get(p)
                if rank is not None and (best_rank is None or rank<best_rank):
                    best_rank=rank;best_pair=p
            if best_pair is None:break
            merged=''.join(best_pair)
            result=[];i=0
            while i<len(syms):
                if i<len(syms)-1 and (syms[i],syms[i+1])==best_pair:
                    result.append(merged);i+=2
                else:
                    result.append(syms[i]);i+=1
            syms=result
        return tuple(syms)

    def encode_len(self,word):
        x=self.pieces(word)
        return len(x) if x is not None else -1

    def to_json(self):
        return {'algorithm':'independent frequency-weighted classical BPE',
            'alphabet':sorted(self.alphabet),'merges':[list(x) for x in self.merges],
            'effective_vocab':len(set(self.alphabet)|{''.join(p) for p in self.merges})+1,
            'vocab_requested':2000,'unknown_symbol':'<unk>','boundary_symbol':'▁'}

    @classmethod
    def from_json(cls,j):
        return cls(j['alphabet'],j['merges'])


def run_unit_tests():
    model=ClassicBPE.fit(['abab']*4+['ac']*3+['abd']*2, vocab_size=12)
    assert model.encode_len('a')>=1
    assert model.encode_len('abc')>=1
    assert model.encode_len('zz')==-1
    assert model.pieces('abab')==model.pieces('abab')
    # Weighted word frequencies and training order should not affect tie-breaking.
    words=['abba','abba','aab','bab','cbd','abba','aaa','bbc']
    a=ClassicBPE.fit(words,vocab_size=16)
    b=ClassicBPE.fit(list(reversed(words)),vocab_size=16)
    assert a.to_json()==b.to_json(), 'order dependence'
    # Serialization must roundtrip without behavior change.
    c=ClassicBPE.from_json(json.loads(json.dumps(a.to_json())))
    assert [a.pieces(w) for w in set(words)]==[c.pieces(w) for w in set(words)]
    # Weighted pair-frequency conservations for a single manual merge.
    tiny=ClassicBPE.fit(['xy']*9+['xz']*3,vocab_size=7)
    assert tiny.merges[0]==('▁','x'),tiny.merges
    return {'toy_training':True,'training_order_invariant':True,'roundtrip':True}

if __name__=='__main__':
    print(run_unit_tests())
