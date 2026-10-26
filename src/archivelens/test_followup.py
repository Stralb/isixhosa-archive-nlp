"""Pre-specified synthetic controls: neither the archive nor historical labels are needed."""
import unittest
import numpy as np
from archivelens.audit import Page, make_units, estimates
from archivelens.followup import (adjust_pvalues,sign_reversal,sensitivity_gap,
    paired_permutation,leave_one_out,shingle_digest,SourceWords,discover_repetitions,
    repeated_mask,recount_unmasked_words)


def page(year,day,pnum,den,count,month=1):
    return Page('IMVO',year,month,f'IMVO_{year}-{month:02}-{day:02}',pnum,50,den,
                np.array([count],dtype=np.int64))


class FollowupTest(unittest.TestCase):
    def test_multiple_comparison_math(self):
        p=np.array([.04,.001,.01])
        np.testing.assert_allclose(adjust_pvalues(p,'bh'),[.04,.003,.015])
        np.testing.assert_allclose(adjust_pvalues(p,'holm'),[.04,.003,.02])
        with self.assertRaises(ValueError): adjust_pvalues(np.array([-.01]),'bh')
    def test_sign_reversal_is_separate_from_magnitude(self):
        x=np.array([[-1,1],[2,2],[-3,-4],[-2,5]],dtype=float)
        self.assertEqual(sign_reversal(x).tolist(),[True,True])
        self.assertEqual(sensitivity_gap(x).tolist(),[3,5])
    def test_month_permutation_never_breaks_issues_or_counts(self):
        a=make_units('IMVO',1885,[page(1885,1,1,100,70),page(1885,1,2,20,0),page(1885,2,1,100,50),page(1885,3,1,70,20,month=2)])
        b=make_units('IMVO',1888,[page(1888,1,1,100,60),page(1888,2,1,120,50),page(1888,3,1,90,20,month=2)])
        original_id=set(a.ids+b.ids)
        for seed in range(42):
            x,y=paired_permutation(a,b,np.random.default_rng(seed))
            self.assertEqual(set(x.ids+y.ids),original_id)
            self.assertEqual(len(x.ids),len(a.ids))
            self.assertEqual(len(y.ids),len(b.ids))
            for m in [1,2]:
                self.assertEqual(len(x.month_indices[m]),len(a.month_indices[m]))
                self.assertEqual(len(y.month_indices[m]),len(b.month_indices[m]))
            for unit in (x,y):
                for id, n in zip(unit.ids,unit.n_pages):
                    if id.endswith('1885-01-01'):
                        self.assertEqual(n,2)
    def test_leave_one_issue_out_changes_weighted_contrast(self):
        a=make_units('IMVO',1885,[page(1885,1,1,100,70),page(1885,1,2,20,0),page(1885,2,1,100,50)])
        b=make_units('IMVO',1888,[page(1888,1,1,100,60),page(1888,2,1,90,70)])
        orig,loo=leave_one_out(a,b)
        self.assertEqual(len(loo),4)
        self.assertEqual(len([r for r in loo if r['issue']=='IMVO_1885-01-01']),1)
        np.testing.assert_allclose(loo[0]['deltas'][0,0]-orig[0,0],loo[0]['change'][0,0])
    def test_repeat_mask_requires_distinct_issue_and_full_thirteen_words(self):
        w=('alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu '
           'omicron').split()
        x=SourceWords(page(1885,1,1,100,0),w)
        y=SourceWords(page(1885,1,2,100,0),w)
        other=SourceWords(page(1885,2,1,100,0),w)
        self.assertEqual(len(discover_repetitions([x,y])),0)
        repeated=discover_repetitions([x,y,other])
        self.assertTrue(len(repeated)>=1)
        mask=repeated_mask(w,repeated)
        self.assertEqual(mask.sum(),len(w))
        tail=w+['different','beyond']
        z=repeated_mask(tail,repeated)
        self.assertTrue(np.all(z[:len(w)]))
        self.assertTrue(np.all(~z[-2:]))
    def test_no_repeat_fixture_and_recount(self):
        w=['aaaa']*40+['bbbb']*15
        r=SourceWords(page(1885,1,1,100,0),w)
        mask=np.zeros(len(w),dtype=bool);mask[:5]=True
        out=recount_unmasked_words(r,mask,{'aaaa':0,'bbbb':1})
        self.assertEqual(out.n_words,50)
        self.assertEqual(out.counts.tolist(),[35,15])
        self.assertEqual(out.denom,50)
        self.assertIsNone(recount_unmasked_words(r,np.ones(len(w),dtype=bool),{'aaaa':0}))


if __name__=='__main__': unittest.main()
