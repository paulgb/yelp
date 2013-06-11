
from mem import cache
from numpy import log, exp

from text_features import stem, vectorizer, tfidf

@cache
def prepare_features(table, max_features, vect=None):
    stemmed_text = stem(table.text)
    if vect is None:
        vect = vectorizer(stemmed_text, max_features)
    features = tfidf(stemmed_text, vect)
    return vect, features

@cache
def prepare_targets(table):
    votes = table.votes_useful
    return log(votes + 1)

@cache
def transform_predictions(predictions):
    return exp(predictions) - 1

