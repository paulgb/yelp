
import config

from load import load_reviews
from pipeline import prepare_features, prepare_targets
from cross_val import split
from model import train_model, predict

def cross_val_model():
    data = load_reviews(config.DATA_ZIP_FILE, config.TRAINING_SET_FILE)

    splits = split(data, config.CV_SPLITS)

    for (train, test) in splits:
        vect, features = prepare_features(train, config.MAX_FEATURES)
        targets = prepare_targets(train)

        model = train_model(features, targets)

        vect, test_features = prepare_features(test, config.MAX_FEATURES, vect)
        predictions = predict(test_features, model)



if __name__ == '__main__':
    cross_val_model()
    

