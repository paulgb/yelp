
from sys import argv
import pickle
import re

import pandas as pd
from nltk.stem.porter import PorterStemmer

class Stemmer(object):
    def __init__(self):
        self.splitter = re.compile("[^\w'`]+")
        self.stemmer = PorterStemmer()

    def stem_sentence(self, sentence):
        words = self.splitter.split(sentence)
        words = map(self.stemmer.stem, words)
        return ' '.join(words)


def stem(infile, outfile):
    reviews = pd.read_csv(infile, parse_dates=[1], na_values=[], keep_default_na=False)
    stemmer = Stemmer()
    stemmed_reviews = reviews.text.map(stemmer.stem_sentence)
    pickle.dump(stemmed_reviews, file(outfile, 'w'))


if __name__ == '__main__':
    stem(*argv[1:])

