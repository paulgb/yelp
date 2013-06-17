
# Data file configuration
DATA_ZIP_FILE = 'data/yelp_training_set.zip'
TRAINING_SET_FILE = 'yelp_training_set/yelp_training_set_review.json'

# Data sampling
SAMPLE_SIZE = 10000

# Cross-validation configuration
CV_SPLITS = 3

# Featurization configuration
MAX_FEATURES = 1000
PCA_FEATURES = 150

# Training
BATCH_SIZE = 400

# Model configuration
HIDDEN_LAYERS = (70,)
ACTIVATION_FUNCTION = 'tanh'
OPTIMIZE = 'sgd'

