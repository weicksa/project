import unittest

from ..tagger.Tagger import *


class Feat_optimize(unittest.TestCase):

	def setUp(self):
		self.path = "project/Data/"
		self.tagger = Tagger()
		self.training = self.tagger.readCoNLL(self.path + "wsj_train.conll09")
		self.develop = self.tagger.readCoNLL(self.path + "wsj_dev.conll09")
		self.test = self.tagger.readCoNLL(self.path + "wsj_test.conll09")
		self.extractor = FeatureExtractors()
		print("done with reading data")

	def test_pipeline(self):

		self.extractor.extractAllFeatures(self.training)
		self.extractor.extractAllFeatures(self.develop)
		self.extractor.extractAllFeatures(self.test)
		print("done with extraction")
		feature_count = 0
		for key in self.extractor.mapper.map:
			feature_count += 1

		class_mapper = self.extractor.class_mapper
		model = Perceptron(class_mapper, feature_count)

		model.train(self.training, 3)

		for sentence in self.develop:
			for token in sentence.tokens:
				model.predict(token)

		for sentence in self.test:
			for token in sentence.tokens:
				model.predict(token)

		print(f"Evaluation and ConfusionMatrix")
		matrix_training = ConfusionMatrix(self.training)
		matrix_test = ConfusionMatrix(self.test)
		matrix_develop = ConfusionMatrix(self.develop)
		accuracy_test = Evaluation.accuracy(self.test)
		accuracy_develop = Evaluation.accuracy(self.develop)
		accuracy_train = Evaluation.accuracy(self.training)
		print(f"ConfusionMatrix of training Data")
		matrix_training.print(10)
		print(f"Accuracy on training data: {accuracy_train}")
		print("")
		print(f"ConfusionMatrix of development Data")
		matrix_develop.print(10)
		print(f"Accuracy on development data: {accuracy_develop}")
		print("")
		print(f"ConfusionMatrix of test Data")
		matrix_test.print(10)
		print(f"accuracy on test data: {accuracy_test}")



if __name__ == "__main__":
	# import sys;sys.argv = ['', 'Test.testName']
	unittest.main()
