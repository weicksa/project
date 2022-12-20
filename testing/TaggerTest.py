'''
Created on Oct 30, 2020

@author: nastasvi
'''
import unittest

from ..tagger.Tagger import *


class TaggerTest(unittest.TestCase):

    def setUp(self):
        self.path = "project/Data/"
        self.tagger = Tagger()

    def testReadCoNLL(self):
        # // test that empty file  produces empty data structures
        ls = self.tagger.readCoNLL(self.path + "file-empty.txt")
        self.assertEqual(0, len(ls), "Empty file should contain no sentences")
        self.assertEqual(0, self.countWords(ls), "Empty file should contain no words")

        # // test that one sentence file produces small data structures
        ls = self.tagger.readCoNLL(self.path + "file-onesent.txt")
        self.assertEqual(1, len(ls), "One sentence file should contain one sentence")
        self.assertEqual(9, self.countWords(ls), "One sentence file should contain nine words")

        # // test robustness on complete tiger train set

        ls = self.tagger.readCoNLL(self.path + "tiger-2.2.train.conll09")
        self.assertEqual(40472, len(ls), "Tiger train file should contain 40472 sentences")
        self.assertEqual(719530, self.countWords(ls), "Tiger train file word count should be 719530")

        # // not relevant yet.
        es = self.tagger.readCoNLL(self.path + "wsj_dev.treetags.conll09")
        Tagger.extractInstances(es, "NN", "NP")

    def countWords(self, ss):
        counter = 0
        for s in ss:
            counter += s.length()

        return counter
"""
    def testPipeline(self):
        pipeline(self.path + "wsj_train.conll09", self.path + "wsj_dev.conll09")
"""

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
