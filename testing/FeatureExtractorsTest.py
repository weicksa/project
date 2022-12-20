'''
Created on Oct 29, 2020

@author: nastasvi
'''
import unittest

from ..tagger.Tagger import Tagger
from ..tagger.model.FeatureExtractors import FeatureExtractors

"""
changed Tagger.readData to Tagger.readCoNLL, because Tagger.readData does not exist
changed self.filename to self.path and changed all file imports and exports accordingly

Added new features - changed test accordingly!!!
"""


class Test(unittest.TestCase):

    def setUp(self):
        self.path = "project/Data/"
        self.tagger = Tagger()

    def testExtractFeatures(self):
        sentences = self.tagger.readCoNLL(self.path + "file-onesent.txt")
        s = sentences[0]
        fes = FeatureExtractors()
        for token in s.tokens:
            fes.extractFeatures(token)
        self.assertEqual(6, len(s.get(0).features))
        self.assertEqual(8, len(s.get(1).features))
        self.assertEqual(9, len(s.get(2).features))
        self.assertEqual(8, len(s.get(3).features))
        self.assertEqual(10, len(s.get(4).features))
        self.assertEqual(7, len(s.get(5).features))
        self.assertEqual(10, len(s.get(6).features))
        self.assertEqual(10, len(s.get(7).features))
        self.assertEqual(6, len(s.get(8).features))
        self.assertEqual(0, self.intersection(s.get(0).features, s.get(1).features))
        self.assertEqual(0, self.intersection(s.get(1).features, s.get(2).features))
        self.assertEqual(1, self.intersection(s.get(2).features, s.get(4).features))
        self.assertEqual(3, self.intersection(s.get(6).features, s.get(7).features))
        self.assertEqual(0, self.intersection(s.get(3).features, s.get(5).features))
        self.assertEqual(0, self.intersection(s.get(2).features, s.get(5).features))
        # print("Features: {}".format(s.get(0).features))

    def testExtractAllFeatures(self):
        ss = self.tagger.readCoNLL(self.path + "file-onesent.txt")
        FeatureExtractors().extractAllFeatures(ss)
        s = ss[0]
        self.assertEqual(6, len(s.get(0).features))
        self.assertEqual(8, len(s.get(1).features))
        self.assertEqual(9, len(s.get(2).features))
        self.assertEqual(8, len(s.get(3).features))
        self.assertEqual(10, len(s.get(4).features))
        self.assertEqual(7, len(s.get(5).features))
        self.assertEqual(10, len(s.get(6).features))
        self.assertEqual(10, len(s.get(7).features))
        self.assertEqual(6, len(s.get(8).features))
        self.assertEqual(0, self.intersection(s.get(0).features, s.get(1).features))
        self.assertEqual(0, self.intersection(s.get(1).features, s.get(2).features))
        self.assertEqual(1, self.intersection(s.get(2).features, s.get(4).features))
        self.assertEqual(3, self.intersection(s.get(6).features, s.get(7).features))
        self.assertEqual(0, self.intersection(s.get(3).features, s.get(5).features))
        self.assertEqual(0, self.intersection(s.get(2).features, s.get(5).features))

    def testWriteToFile(self):
        ss = self.tagger.readCoNLL(self.path + "file-onesent.txt")
        fes = FeatureExtractors()
        fes.extractAllFeatures(ss)
        fes.writeToFile(ss, self.path + "file-onesent.svmmulti")
        fes.readFromFile(self.path + "file-onesent.svmmulti")

    def testReadFromFile(self):

        ss = self.tagger.readCoNLL(self.path + "tiger-2.2.train.conll09")
        fes = FeatureExtractors()
        fes.extractAllFeatures(ss)
        fes.writeToFile(ss, self.path + "file-tiger.svmmulti")
        ss = fes.readFromFile(self.path + "file-tiger.svmmulti")
        #        # // test robustness on complete tiger train set
        print(f"test: {len(ss[-1].tokens)}")

        # test read from file for smaller file
        one_sent = self.tagger.readCoNLL(self.path+ "file-onesent.txt")
        one_fes = FeatureExtractors()
        one_fes.extractAllFeatures(one_sent)
        fes.writeToFile(one_sent, self.path + "file-onesent.svmmulti")
        one_sent = one_fes.readFromFile(self.path + "file-onesent.svmmulti")
        self.assertEqual(1, len(one_sent))
        self.assertEqual(9, self.countWords(one_sent))
        # end of new test

        self.assertEqual(40472, len(ss), "Tiger train file should contain 40472 sentences")
        self.assertEqual(719530, self.countWords(ss), "Tiger train file word count should be 719530")

    def intersection(self, array1, array2):
        return len(set(array1).intersection(array2))

    def countWords(self, sentences):
        return sum([s.length() for s in sentences])


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testExtractFeatures']
    unittest.main()
