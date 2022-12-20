'''
Created on Oct 28, 2020

@author: nastasvi
'''


class Token(object):
    '''
    classdocs
    '''

    def __init__(self, word=None, label=None, prediction=None, correctLabelIndex=None, predictedLabelIndex=None,
                 features=[], previous=None, previous_2=None, previous_3=None, next=None, next_2=None, next_3=None,
                 length_prev=None, length_next=None):
        '''
        Constructor
        '''
        self.word = word
        self.label = label
        self.prediction = prediction

        self.correctLabelIndex = correctLabelIndex
        self.predictedLabelIndex = predictedLabelIndex

        self.features = features

        self.previous = previous
        self.previous_2 = previous_2
        self.previous_3 = previous_3
        self.next = next
        self.next_2 = next_2
        self.next_3 = next_3
        try:
            self.length = len(self.word)
            self.first_letter = self.word[0]
        except TypeError:
            self.length = None
            self.first_letter = None
        self.length_prev = length_prev
        self.length_next = length_next

    def toString(self):
        word_str = "None"

        if self.word != None:
            word_str = "W: " + self.word
            word_str += "\tL: " + Token.getVal(self.label) + "(" + Token.getVal(self.correctLabelIndex) + ")"
            word_str += "\tPL: " + Token.getVal(self.word.prediction) + "(" + Token.getVal(
                self.predictedLabelIndex) + ")"

        return word_str

    @staticmethod
    def getVal(val):
        if val == None:
            return "None"
        return "{}".format(val)
