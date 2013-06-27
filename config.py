
# Data file configuration
DATA_ZIP_FILE = 'data/yelp_training_set.zip'
TRAINING_SET_FILE = 'yelp_training_set/yelp_training_set_review.json'
VAL_ZIP_FILE = 'data/yelp_test_set.zip'
VAL_SET_FILE = 'yelp_test_set/yelp_test_set_review.json'
OUTPUT_FILE = 'out.csv'

# Data sampling
SAMPLE_SIZE = False

# Cross-validation configuration
CV_SPLITS = 3

# Featurization configuration
TEXT_FEATURES = 1000
PCA_FEATURES = 150
USE_SCALE = False

# Training
BATCH_SIZE = 400

# Model configuration
HIDDEN_LAYERS = (60,)
ACTIVATION_FUNCTION = 'norm:std+tanh'
OPTIMIZE = 'hf' # hf or sgd
WEIGHT_L2 = 1.2

