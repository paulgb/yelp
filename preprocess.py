
from sys import argv
import pickle

import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

MAX_FEATURES = 25


def preprocess(infile, outfile):
    reviews = pd.read_csv(infile, parse_dates=[1], na_values=[], keep_default_na=False)
    reviews = reviews.irow(range(1000))
    stemmer = PorterStemmer()
    vectorizer = TfidfVectorizer(stop_words='english', max_features=MAX_FEATURES, preprocessor=stemmer.stem)
    npy.save(r
    print vectorizer.fit_transform(reviews.text)
    

if __name__ == '__main__':
    preprocess(*argv[1:])

