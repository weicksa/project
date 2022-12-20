'''
Created on Oct 28, 2020

@author: nastasvi
'''


class Sentence(object):
    '''
    classdocs
    '''

    def __init__(self, tokens=[]):
        '''
        Constructor
        '''
        self.tokens = tokens

    def length(self)-> int:
        return len(self.tokens)

    def isEmpty(self):
        return len(self.tokens) == 0

    def add(self, token):
        self.tokens.append(token)

    def get(self, i):
        return self.tokens[i]

    def toString(self):
        sent = ""
        for x in self.tokens:
            # sent += "{} ({})\t".format(x.word, x.features)
            sent += "{} ".format(x.word)
        return sent
