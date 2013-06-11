
import config

from sklearn.metrics import mean_squared_error
from load import load_reviews
from pipeline import prepare_features, prepare_targets, transform_predictions
from cross_val import split
from model import train_model, predict

from mem import cache

@cache
def train_and_test(train, test):
    vect, features = prepare_features(train, config.MAX_FEATURES)
    targets = prepare_targets(train)

    model = train_model(features, targets)

    vect, test_features = prepare_features(test, config.MAX_FEATURES, vect)
    predictions = predict(test_features, model)

    #predictions = transform_predictions(results)


    return mean_squared_error(predictions, prepare_targets(test))

def cross_val_model():
    data = load_reviews(config.DATA_ZIP_FILE, config.TRAINING_SET_FILE)

    splits = split(data, config.CV_SPLITS)
    errors = list()

    for (train, test) in splits:
        errors.append(train_and_test(train, test))

    return errors


if __name__ == '__main__':
    errors = cross_val_model()
    print 'Errors: %s' % errors
    print 'Avg: %s' % (sum(errors) / float(len(errors)))
    

