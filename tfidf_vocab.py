
from sys import argv
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

MAX_FEATURES = 1000

def tfidf_vocab(infile, outfile):
    stemmed_text = pickle.load(file(infile))
    vectorizer = TfidfVectorizer(stop_words='english', max_features=MAX_FEATURES)
    vectorizer.fit(stemmed_text)
    pickle.dump(vectorizer, file(outfile, 'w'))
    

if __name__ == '__main__':
    tfidf_vocab(*argv[1:])

