
from mem import cache
from numpy import log, exp
from sklearn.decomposition import PCA

from text_features import stem, vectorizer, tfidf

@cache
def prepare_features(table, max_features, vect=None, pca=None):
    stemmed_text = stem(table.text)
    if vect is None:
        vect = vectorizer(stemmed_text, max_features)
    features = tfidf(stemmed_text, vect)
    if hasattr(features, 'toarray'):
        features = features.toarray()
    if pca is None:
        pca = PCA(150).fit(features)
    features = pca.transform(features)
    return vect, pca, features

@cache
def prepare_targets(table, scale=None):
    votes = log(table.votes_useful + 1)
    print 'min', min(votes)
    if scale is None:
        scale = 1
        #scale = 1 / max(votes)
    return scale, votes * scale

@cache
def transform_predictions(predictions):
    return exp(predictions) - 1

