
from sys import argv
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

MAX_FEATURES = 1000

def tfidf(vec_file, infile, outfile):
    vectorizer = pickle.load(file(vec_file))
    stemmed_text = pickle.load(file(infile))

    out = vectorizer.transform(stemmed_text)
    pickle.dump(out, file(outfile, 'w'))

if __name__ == '__main__':
    tfidf(*argv[1:])

