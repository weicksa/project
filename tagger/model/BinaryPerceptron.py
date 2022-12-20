from ..data.Token import Token
from ..data.Sentence import Sentence
from ..data.StringMapper import StringMapper
from ..model.BinaryWeights import BinaryWeights
import random


class BinaryPerceptron(object):

    def __init__(self, num):
        self.weights = BinaryWeights(num)

    def predict(self, token):
        # get the score of token
        pred = self.weights.score(token.features)

        # round score to either one or zero, depending on what the value of pred
        # is closer to
        if 1-pred > 0+pred:
            prediction = 0
        else:
            prediction = 1

        # save prediction in token and return the prediction
        token.predictedLabelIndex = prediction
        return prediction

    def train(self, trainingData, numberOfIterations):
        for i in range(numberOfIterations + 1):
            # randomly iterate over training data and the tokens in the training data
            for a in random.sample(range(len(trainingData)), len(trainingData)):
                for b in random.sample(list(range(trainingData[a].length())), trainingData[a].length()):
                    token = trainingData[a].tokens[b]
                    pred = self.predict(token)
                    # check whether our prediction is right, if it is not
                    # check whether our prediction is too big or to small
                    # if too big, update the weights to be smaller
                    # else update them to be bigger
                    if pred != float(token.correctLabelIndex):
                        if pred > token.correctLabelIndex:
                            self.weights.update(token.features, -1)
                        else:
                            self.weights.update(token.features, 1)
