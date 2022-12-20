from ..data.Token import Token
from ..data.Sentence import Sentence
from ..data.StringMapper import StringMapper
import random


class Weights(object):
    def __init__(self, class_mapper: StringMapper, numFeatures):
        # initialize class map with bias = 1.0
        self.class_map = {}
        # create initial entries in class map, based on numClasses and numFeatures
        for key in class_mapper.map.keys():
            if key not in self.class_map:
                self.class_map[class_mapper.map[key]] = {"bias": 1.0}
            for b in range(numFeatures):
                # initialize weights with random values (attempt to fix overfitting to class 1)
                self.class_map[class_mapper.map[key]][b] = 0

    def score(self, classID, featureVector):
        # compute score as scalar for a specific class for a feature vector
        scalar = 0
        for feat in featureVector:
            if feat not in self.class_map[classID]:
                # self.class_map[classID][feat] = 1
                # scalar += 1 * self.class_map[classID][feat]
                scalar += 0
            else:
                scalar += 1 * self.class_map[classID][feat]
        # always add the bias value to score
        scalar += 1 * self.class_map[classID]["bias"]
        return scalar

    def update(self, prediction, correctLabelIndex, features, learningRate):
        # always set bias to learningRate
        for key in self.class_map.keys():
            self.class_map[key]["bias"] = 1.0

        # if the prediction is not correct, decrease weights for wrongly predicted class
        # and increase weights for correct class
        if prediction is not correctLabelIndex:
            for al in features:
                if al in self.class_map[prediction]:
                    self.class_map[prediction][al] += (1 * learningRate)
                else:
                    self.class_map[prediction][al] = (1 * learningRate)

            for el in features:
                if el in self.class_map[correctLabelIndex]:
                    self.class_map[correctLabelIndex][el] += ((-1) * learningRate)
                else:
                    self.class_map[correctLabelIndex][el] = ((-1) * learningRate)
        """
        else:
            for al in features:
                if al in self.class_map[prediction]:
                    self.class_map[prediction][al] += (1 * learningRate)
                else:
                    self.class_map[prediction][al] = (1 * learningRate)
        """



