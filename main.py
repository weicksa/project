from Data import *
from tagger.Tagger import pipeline


if __name__ == "__main__":
    pipeline("Data/wsj_train.conll09", "Data/wsj_dev.conll09")