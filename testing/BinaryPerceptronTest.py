'''
Created on Oct 29, 2020

@author: nastasvi
'''
import unittest

from ..tagger.model.BinaryWeights import BinaryWeights
from ..tagger.model.BinaryPerceptron import BinaryPerceptron

from ..tagger.data.Sentence import Sentence
from ..tagger.data.Token import Token


class BinaryPerceptronTest(unittest.TestCase):

    def testWeights(self):
        weights = BinaryWeights(5)

        # // first, try to update the weights with a feature vector
        featureVector1 = [1, 2, 3]
        weights.update(featureVector1, 0.5)

        score1 = weights.score(featureVector1)
        self.assertAlmostEqual(2, score1, 2, "Score should be 4 * 0.5 = 2")

        # // then score a smaller feature vector
        featureVector2 = [1, 2]
        score2 = weights.score(featureVector2)
        self.assertAlmostEqual(1.5, score2, 2, "Score should be 3 * 0.5 = 1.5")

        # // now try to score another feature vector
        # // now with unseen feature (from "test set")
        # // how to treat?
        featureVector3 = [1, 2, 99]
        score3 = weights.score(featureVector3)
        self.assertAlmostEqual(1.5, score3, 2, "Score should be 3 * 0.5 = 1.5")

    def testBinaryPerceptron(self):
        perceptron = BinaryPerceptron(6)
        trainSentenceList = self.generateTrainSentences()

        perceptron.train(trainSentenceList, 3)
        print("{}".format(perceptron.weights.weights))
        # // make a prediction for the first token
        prediction1 = perceptron.predict(trainSentenceList[0].get(0))
        self.assertEqual(prediction1, 1, "Prediction for the first token should be 1")

        # // make a prediction for the last token
        prediction4 = perceptron.predict(trainSentenceList[0].get(3))
        self.assertEqual(prediction4, 0, "Prediction for the fourth token should be 0")

        # // make a prediction for an unseen token
        testToken = self.generateTestToken()
        predictionTest = perceptron.predict(testToken)
        self.assertEqual(predictionTest, 0, "Prediction for the unseen token should be 0")
        self.assertEqual(predictionTest, testToken.predictedLabelIndex, "Prediction should be saved in token!")

    def generateTrainSentences(self):
        testSentenceList = []
        testSentence = Sentence()
        testSentenceList.append(testSentence)

        token1 = Token()
        token1.features = [1, 2, 3]
        token1.correctLabelIndex = 1
        testSentence.add(token1)

        token2 = Token()
        token2.features = [2, 3]
        token2.correctLabelIndex = 1
        testSentence.add(token2)

        token3 = Token()
        token3.features = [4, 5, 6]
        token3.correctLabelIndex = 0
        testSentence.add(token3)

        token4 = Token()
        token4.features = [5, 6]
        token4.correctLabelIndex = 0
        testSentence.add(token4)

        return testSentenceList

    def generateTestToken(self):
        testToken = Token()
        testToken.correctLabelIndex = 1
        testToken.features = [1, 5, 6]
        return testToken


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
