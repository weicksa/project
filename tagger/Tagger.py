'''
Created on Oct 29, 2020

@author: nastasvi
'''

from .data.Token import Token
from .data.Sentence import Sentence
from .data.StringMapper import StringMapper
from .model.FeatureExtractors import FeatureExtractors
from .data.ConfusionMatrix import ConfusionMatrix
from .model.Perceptron import Perceptron
from .model.Evaluation import Evaluation


def pipeline(train, dev):
    feature_count = 0
    class_count = 0
    tagger = Tagger()
    extractor = FeatureExtractors()

    training = tagger.readCoNLL(train)
    develop = tagger.readCoNLL(dev)
    print("done with reading data")

    extractor.extractAllFeatures(training)
    extractor.extractAllFeatures(develop)
    print("done with extraction")
    for key in extractor.mapper.map:
        feature_count += 1

    extractor.writeToFile(training, "project/Data/file_training.svmmulti")
    class_mapper = extractor.class_mapper
    model = Perceptron(class_mapper, feature_count)
    print("now training model")
    model.train(training, 3)

    print("now making predictions")
    for sentence in develop:
        for token in sentence.tokens:
            model.predict(token)

    print("now doing evaluation and ConfusionMatrix")
    matrix = ConfusionMatrix(develop)
    accuracy = Evaluation.accuracy(develop)
    accuracy_train = Evaluation.accuracy(training)

    matrix.print(10)
    print(f"accuracy on development data: {accuracy}")
    print(f"Accuracy on training data: {accuracy_train}")



class Tagger(object):
    '''
    classdocs
    '''

    def __init__(self):
        self.class_map = StringMapper()

    def readCoNLL(self, filename):
        self.class_map
        res_list = []
        i = 1
        token_counter = 0
        with open(filename) as source:
            lines = source.readlines()
            length = len(lines) - 1
            sent_list = []
        for line in lines:
            length -= 1
            spl = line.split()
            try:
                # check wether current line still belongs to the same sentence
                if len(spl) > 0:
                    tok = Token(word=spl[1], label=spl[4], prediction=spl[5],
                                correctLabelIndex=self.class_map.lookup(spl[4]))
                    sent_list.append(tok)
                # if current line is a new sentence, complete the old sentence
                else:
                    token_counter += len(sent_list)
                    copy_sent_list = sent_list.copy()
                    sent = Sentence(tokens=copy_sent_list)
                    res_list.append(sent)

                    # append current token
                    sent_list.clear()
                    i += 1
                    tok = Token(word=spl[1], label=spl[4], prediction=spl[5])
                    sent_list.append(tok)
            except IndexError:
                pass

        # initialize the previous and next values for all tokens
        for sent in res_list:
            for i in range(sent.length()):
                current_token = sent.get(i)
                try:
                    current_token.previous = sent.get(i - 1).word
                    current_token.previous_2 = sent.get(i-2).word
                    current_token.previous_3 = sent.get(i-3).word
                except IndexError:
                    current_token.previous = None
                    current_token.previous_2 = None
                    current_token.previous_3 = None
                try:
                    current_token.next = sent.get(i + 1).word
                    current_token.next_2 = sent.get(i+2).word
                    current_token.next_3 = sent.get(i+3).word
                except IndexError:
                    current_token.next = None
                    current_token.next_2 = None
                    current_token.next_3 = None

        return res_list

    @staticmethod
    def extractInstances(data, goldLabel, predLabel):
        # extract specific Instances from our data, with the specified gold and pred labels,
        # also prints the context
        for sentence in data:
            for i in range(sentence.length()):
                token = sentence.get(i)
                if token.label == goldLabel and token.prediction == predLabel:
                    try:
                        for a in range(i - 3, i):
                            print(f"{sentence.get(a).word}\t{sentence.get(a).label}\t{sentence.get(a).prediction}")
                    except IndexError:
                        pass
                    print(f"*{token.word}*\t{token.label}\t{token.prediction}")
                    try:
                        for b in range(i + 1, i + 4):
                            print(f"{sentence.get(b).word}\t{sentence.get(b).label}\t{sentence.get(b).prediction}")
                    except IndexError:
                        pass
                    print("*******************")



    # this is a comment made by Sandro

    # comment by tana
