from ..data.Token import Token
from ..data.Sentence import Sentence
from ..data.StringMapper import StringMapper
from ..model.Weights import Weights
import random


class Perceptron(object):

	def __init__(self, class_mapper: StringMapper, numFeatures):
		self.weights = Weights(class_mapper, numFeatures)
		self.class_mapper = class_mapper

	def predict(self, token):
		# make a prediction for a tokens class based on the scoring function
		res_dict = {}
		# iterate over all known classes and compute the scalar for each one
		# append them to res_list in a tuple with (predicted_class, score)
		for cl in self.class_mapper.map.keys():
			pred = self.weights.score(self.class_mapper.lookup(cl), token.features)
			if pred in res_dict:
				res_dict[pred].append(self.class_mapper.lookup(cl))
			else:
				res_dict[pred] = [self.class_mapper.lookup(cl)]
			# res_list.append((self.class_mapper.lookup(cl), pred))

		# sort the results by score
		sort_res = sorted(res_dict.keys(), reverse=True)
		prediction = res_dict[sort_res[0]][random.randint(0, len(res_dict[sort_res[0]]) - 1)]
		# the final prediction is the class with the highest score
		token.predictedLabelIndex = prediction
		token.prediction = self.class_mapper.inverseLookup(prediction)
		# return list of all results with equal score and append score at the end for indexing
		res = res_dict[sort_res[0]]
		res.append(sort_res[0])
		return res

	def train(self, trainingData, numberOfIterations):

		for i in range(numberOfIterations + 1):
			# randomly iterate over sentences and tokens
			for a in random.sample(range(len(trainingData)), len(trainingData)):
				for b in random.sample(list(range(trainingData[a].length())), trainingData[a].length()):
					token = trainingData[a].tokens[b]
					pred = self.predict(token)
					# check wether the prediction is correct,
					# if it's too small , increase weights for the right class and decrease them for the
					# wrong prediction
					# if it's too big, decrease weights for the right class and increase them for
					# the wrong prediction

					# iterate over all classes with the same score, do not use last element in pred, because its
					# the score
					for cl in pred[:-1]:
						if cl != token.correctLabelIndex:
							# if the predicted class is wrong, lower its score, else increase it
							self.weights.update(cl, token.correctLabelIndex, token.features, -1.0)

						else:
							self.weights.update(cl, token.correctLabelIndex, token.features, 1.0)
