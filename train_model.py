#!/usr/bin/env python

import config

from sklearn.metrics import mean_squared_error
from load import load_reviews
from pipeline import prepare_features, prepare_targets, transform_predictions
from cross_val import split
from model import train_model, predict

from numpy import array
from mem import cache

#@cache
def train_and_test(train, test):
    vect, features = prepare_features(train, config.MAX_FEATURES)
    scale, targets = prepare_targets(train)
    print 'scale:', scale

    vect, test_features = prepare_features(test, config.MAX_FEATURES, vect)
    scale, test_targets = prepare_targets(test, scale)

    model = train_model(features, targets, test_features, test_targets, config.HIDDEN_LAYERS)
    predictions = predict(test_features, model)
    print predictions
    print array(test_targets)
    
    mse = mean_squared_error(predictions / scale, test_targets / scale)
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
    

