
import lmj.nn as nn
import lmj.cli as cli

from numpy import matrix

from mem import cache

cli.enable_default_logging()

@cache
def train_model(features, targets, test_features, test_targets, hidden_layers):
    n_features = features.shape[1]

    layers = (n_features,) + hidden_layers + (1,)
    print 'layers', layers
    ex = nn.Experiment(nn.Regressor, layers=layers)

    features = features.toarray()
    targets = matrix(targets).T
    test_features = test_features.toarray()
    test_targets = matrix(test_targets).T

    ex.run((features, targets), (test_features, test_targets))

    return ex.network

@cache
def predict(features, model):
    features = features.toarray()

    predictions = model(features)
    print 'shape', predictions.shape
    return predictions[:,0]
    
