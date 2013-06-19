
from mem import cache
from numpy import log, exp
from sklearn.decomposition import PCA

from sklearn.feature_extraction.text import TfidfVectorizer
from text_features import stem

class Pipeline:
    def __init__(self, max_features):
        self.max_features = max_features


    def fit(self, features):
        stemmed_text = stem(features.text)

        self.tfidf = TfidfVectorizer(stop_words='english', max_features=self.max_features)
        features = self.tfidf.fit_transform(stemmed_text)

        features = features.toarray()

        self.pca = PCA(150).fit(features)
        

    def transform(self, features):
        stemmed_text = stem(features.text)

        features = self.tfidf.transform(stemmed_text)
        features = features.toarray()

        return self.pca.transform(features)


    def transform_targets(self, table):
        votes = log(table.votes_useful + 1)
        #print 'min', min(votes)
        return votes


    def transform_predictions(self, predictions):
        return exp(predictions) - 1

