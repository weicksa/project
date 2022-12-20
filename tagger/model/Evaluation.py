'''
Created on Oct 29, 2020

@author: nastasvi
'''


class Evaluation(object):
    '''
    classdocs
    '''

    @staticmethod
    def accuracy(data):
        correct = 0
        total = 0
        for sentence in data:
            for tok in sentence.tokens:
                total += 1
                if tok.label == tok.prediction:
                    correct += 1

        return correct / total
