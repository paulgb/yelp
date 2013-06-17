
import lmj.nn as nn
import lmj.cli as cli

from numpy import matrix

from mem import cache

cli.enable_default_logging()

@cache
def train_model(features, targets,
        test_features, test_targets,
        hidden_layers, batch_size,
        activation, optimize):
    n_features = features.shape[1]

    layers = (n_features,) + hidden_layers + (1,)

    ex = nn.Experiment(nn.Regressor,
            layers=layers,
            batch_size=batch_size,
            activation=activation,
            optimize=optimize)

    targets = matrix(targets).T

    test_targets = matrix(test_targets).T

    ex.run((features, targets), (test_features, test_targets))

    return ex.network

@cache
def predict(features, model):
    predictions = model(features)
    return predictions[:,0]
    
