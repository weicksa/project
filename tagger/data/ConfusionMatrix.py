from .Sentence import *
from .Token import *


class ConfusionMatrix(object):

    def __init__(self, data):
        # initialize the ConfusionMatrix, it is a Dictionary in a Dictionary
        # the outer dictionary contains the gold_labels, the inner dictionary contains
        # prediction_labels and the corresponding counts
        self.data = data
        outer_dict = {}
        for sentence in data:
            for token in sentence.tokens:
                if token.label in outer_dict:
                    if token.prediction in outer_dict[token.label]:
                        outer_dict[token.label][token.prediction] += 1
                    else:
                        outer_dict[token.label][token.prediction] = 1
                else:
                    outer_dict[token.label] = {}
                    outer_dict[token.label][token.prediction] = 1
        self.matrix = outer_dict

        tag_value_list = []
        for key in outer_dict:
            try:
                tag_value_list.append((key, outer_dict[key][key]))
            except KeyError:
                pass
        self.sort = sorted(tag_value_list, key=lambda value: value[1], reverse=True)

    def numberErrors(self, goldLabel: str, predLabel: str) -> int:
        try:
            return self.matrix[goldLabel][predLabel]
        except KeyError:
            return 0

    def print(self, maxDim: int):
        # print the matrix in a sensible format
        labels = []
        labels.append("  ")
        labels.extend([key for key, val in self.sort[:maxDim]])
        label_helper = labels.copy()
        label_format = []
        for label in label_helper:
            label_format.append(label + ((4 - len(label))*" "))
        print(label_format)
        vals = []
        for el in labels[1:]:
            vals.append(el + ((4-len(el)) * " "))
            for lab in labels[1:]:
                try:
                    vals.append(str(self.matrix[el][lab]) + ((4 - len(str(self.matrix[el][lab]))) * " "))
                except KeyError:
                    vals.append(str(0) + (3 * " "))
            print(f"{vals}")
            vals.clear()

        """
        for gold in labels[1:]:
            for pred in labels[1:]:
                print(f"gold:{gold} pred:{pred} errors: {self.numberErrors(gold, pred)}")
        """



