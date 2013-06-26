
import lmj.nn as nn
import lmj.cli as cli

from numpy import matrix

cli.enable_default_logging()

class NeuralNetModel:
    def __init__(self, hidden_layers, batch_size, activation, optimize, weight_l2):
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.activation = activation
        self.optimize = optimize
        self.weight_l2 = weight_l2
        self.network = None

    def train(self, features, targets, test_features, test_targets):
        n_features = features.shape[1]

        layers = (n_features,) + self.hidden_layers + (1,)

        ex = nn.Experiment(nn.Regressor,
                layers=layers,
                batch_size=self.batch_size,
                activation=self.activation,
                optimize=self.optimize,
                weight_l2=self.weight_l2)

        targets = matrix(targets).T
        test_targets = matrix(test_targets).T

        ex.run((features, targets), (test_features, test_targets))
        self.network = ex.network

        return self

    def predict(self, features):
        predictions = self.network(features)
        return predictions[:,0]

