from ..data.Token import Token
from ..data.Sentence import Sentence
from ..data.StringMapper import StringMapper


class BinaryWeights(object):

    def __init__(self, integer):
        self.weights = {0: 1.0}
        #

    def score(self, featureVector):
        scalar = 0
        # iterate over featureVector, compute score based on scalar
        # with all feature-weight pairs
        for el in featureVector:
            if el not in self.weights:
                pass
            else:
                scalar += 1 * self.weights[el]
        # special case for bias, is always added, independent of features
        scalar += 1 * self.weights[0]
        return scalar

    def update(self, featureVector, learningRate):
        # set bias to learning rate
        self.weights[0] = learningRate

        # iterate over features, if there already is a weight for el,
        # add learning rate, if there is not, initialize it with the value of
        # the learningRate
        for el in featureVector:
            if el in self.weights:
                self.weights[el] += learningRate
            else:
                self.weights[el] = learningRate
