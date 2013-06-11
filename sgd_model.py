
from sklearn.linear_model import SGDRegressor
from sklearn.dummy import DummyRegressor

from mem import cache

@cache
def train_model(features, targets, *params):
    model = SGDRegressor()
    model.fit(features, targets)
    return model

@cache
def predict(features, model):
    return model.predict(features)
    
