
# Data file configuration
DATA_ZIP_FILE = 'data/yelp_training_set.zip'
TRAINING_SET_FILE = 'yelp_training_set/yelp_training_set_review.json'

# Data sampling
SAMPLE_SIZE = None

# Cross-validation configuration
CV_SPLITS = 3

# Featurization configuration
MAX_FEATURES = 1000

# Training
BATCH_SIZE = 400

# Model configuration
HIDDEN_LAYERS = (500,100)
ACTIVATION_FUNCTION = 'tanh'
OPTIMIZE = 'sgd'

