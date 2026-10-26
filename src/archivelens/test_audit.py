"""Synthetic methodological controls for ArchiveLens; no archive file required."""
import unittest
import numpy as np
from archivelens.audit import Page,make_units,estimates,resample_estimates,delta_support,word_grams

def row(year,day,positions,hits,month=1,page_no=2):
    return Page("IMVO",year,month,f"IMVO_{year}-{month:02}-{day:02}",page_no,
                40,positions,np.array([hits],dtype=np.int64))

class ArchiveLensSyntheticTests(unittest.TestCase):
    def test_known_sign_reversal(self):
        earlier=make_units("IMVO",1885,[row(1885,1,100,80),row(1885,2,10,0)])
        later=make_units("IMVO",1888,[row(1888,1,100,70),row(1888,2,10,9)])
        d=(estimates(later,[1])-estimates(earlier,[1]))[:,0]
        self.assertLess(d[0],0)
        self.assertTrue(all(x>0 for x in d[1:]))
        self.assertAlmostEqual(d[0],(79/110-80/110)*10000,places=8)
        self.assertAlmostEqual(d[1],(.8-.4)*10000,places=8)
    def test_proportional_pages_are_stable(self):
        earlier=make_units("IMVO",1885,[row(1885,1,100,80),row(1885,2,10,8)])
        later=make_units("IMVO",1888,[row(1888,1,100,70),row(1888,2,10,7)])
        np.testing.assert_allclose((estimates(later,[1])-estimates(earlier,[1]))[:,0],
                                   [-1000]*4,atol=1e-8,rtol=0)
    def test_publication_issues_not_pages_are_resampled(self):
        one=make_units("IMVO",1885,[row(1885,1,100,80,page_no=1),
                                   row(1885,1,10,0,page_no=2)])
        expected=estimates(one,[1])
        for seed in range(35):
            np.testing.assert_allclose(
                resample_estimates(one,np.array([1]),np.random.default_rng(seed)),
                expected,rtol=0,atol=0)
        self.assertNotEqual(expected[0,0],expected[1,0])
    def test_within_word_overlapping_fourgrams(self):
        n,pos,cnt,repl=word_grams(b"aaaaa bbbb",{"aaaa":0,"bbbb":1})
        self.assertEqual((n,pos,repl),(2,3,0))
        self.assertEqual(cnt.tolist(),[2,1])
    def test_interval_support_is_not_point_sign(self):
        self.assertTrue(delta_support(.3,.1,.6))
        self.assertTrue(delta_support(-.3,-.6,-.1))
        self.assertFalse(delta_support(.3,-.1,.7))
        self.assertFalse(delta_support(-.2,-.3,0.0))

if __name__=="__main__":
    unittest.main()
