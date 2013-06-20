
from numpy import log, exp
from sklearn.decomposition import PCA

from sklearn.feature_extraction.text import TfidfVectorizer
from text_features import stem
from numpy import hstack

from category_average import CategoryAverage

class Pipeline:
    def __init__(self, max_features, use_scale):
        self.max_features = max_features
        self.use_scale = use_scale

    def fit(self, table):

        stemmed_text = stem(table.text)

        self.tfidf = TfidfVectorizer(stop_words='english', max_features=self.max_features)
        text_features = self.tfidf.fit_transform(stemmed_text)
        text_features = text_features.toarray()

        self.pca = PCA(150).fit(text_features)

        self.avg_user = CategoryAverage()
        self.avg_user = self.avg_user.fit(table.user_id, table.votes_useful)

        # scale for votes
        votes = log(table.votes_useful + 1)
        if self.use_scale:
            self.scale = 1 / max(votes)
        else:
            self.scale = 1

        return self
        

    def transform(self, features):
        stemmed_text = stem(features.text)

        text_features = self.tfidf.transform(stemmed_text)
        text_features = text_features.toarray()

        text_features_pca = self.pca.transform(text_features)

        avg_user = self.avg_user.transform(features.user_id)

        return hstack((text_features_pca, avg_user))


    def transform_targets(self, table):
        votes = log(table.votes_useful + 1)
        return votes * self.scale


    def transform_predictions(self, predictions):
        return exp(predictions) - 1

