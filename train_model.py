#!/usr/bin/env python

import config

from sklearn.metrics import mean_squared_error
from load import load_reviews
from pipeline import Pipeline
from cross_val import split
from model import NeuralNetModel

from numpy import array
from mem import cache

def train_and_test(train, test):
    pipeline = Pipeline(config.TEXT_FEATURES, config.USE_SCALE)

    pipeline = cache(pipeline.fit)(train)
    features = cache(pipeline.transform)(train)
    targets = cache(pipeline.transform_targets)(train)

    test_features = cache(pipeline.transform)(test)
    test_targets = cache(pipeline.transform_targets)(test)

    model = NeuralNetModel(config.HIDDEN_LAYERS,
            config.BATCH_SIZE, config.ACTIVATION_FUNCTION, config.OPTIMIZE,
            config.WEIGHT_L2)
    model = cache(model.train)(features, targets, test_features, test_targets)

    predictions = cache(model.predict)(test_features)
    print predictions, test_targets
    
    print pipeline.scale
    print 'err1: ', mean_squared_error(predictions, test_targets)
    mse = mean_squared_error(predictions / pipeline.scale, test_targets / pipeline.scale)
    print 'error: ', mse

    return mse


def cross_val_model():
    data = load_reviews(config.DATA_ZIP_FILE, config.TRAINING_SET_FILE, config.SAMPLE_SIZE)

    splits = split(data, config.CV_SPLITS)
    errors = list()

    for (train, test) in splits:
        errors.append(train_and_test(train, test))

    return errors


if __name__ == '__main__':
    errors = cross_val_model()
    print 'Errors: %s' % errors
    print 'Avg: %s' % (sum(errors) / float(len(errors)))
    

