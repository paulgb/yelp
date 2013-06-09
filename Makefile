
# extract the json data from zip file
training_set_json :
	unzip source_data/yelp_training_set.zip
	mv yelp_training_set training_set_json

# convert the json data to csv
training_set_csv : training_set_json convert.py
	cd training_set_json; python ../convert.py
	mkdir -p training_set_csv
	mv training_set_json/*.csv training_set_csv/

# stem the review text
stemmed/training_set_reviews.dat : stem.py training_set_csv
	mkdir -p stemmed
	python stem.py training_set_csv/yelp_training_set_review.csv stemmed/training_set_reviews.dat

# "train" a tf/idf transformer
tfidf/vocab.dat : training_set_csv tfidf_vocab.py stemmed/training_set_reviews.dat
	mkdir -p tfidf
	python tfidf_vocab.py stemmed/training_set_reviews.dat tfidf/vocab.dat

# extract term frequencies from training data
tfidf/training_set.dat : tfidf/vocab.dat tfidf.py
	python tfidf.py tfidf/vocab.dat stemmed/training_set_reviews.dat tfidf/training_set.dat

# extract target values
target/training_set.dat : training_set_csv extract_target.py
	mkdir -p target
	python extract_target.py training_set_csv/yelp_training_set_review.csv target/training_set.dat

# create cross-validation sets
cv/splits.dat : target/training_set.dat tfidf/training_set.dat cross_val_split.py
	mkdir -p cv
	python cross_val_split.py cv/splits.dat target/training_set.dat tfidf/training_set.dat

# split target
cv/target : cv/splits.dat target/training_set.dat split_data.py
	mkdir -p cv/train/target/
	mkdir -p cv/test/target/
	cd cv; python ../split_data.py splits.dat ../target/training_set.dat target/target
	touch cv/target

# split tfidf
cv/tfidf : cv/splits.dat tfidf/training_set.dat split_data.py
	mkdir -p cv/train/tfidf/
	mkdir -p cv/test/tfidf/
	cd cv; python ../split_data.py splits.dat ../tfidf/training_set.dat tfidf/tfidf
	touch cv/tfidf

# train models
cv/models : cv/tfidf cv/target
	mkdir -p cv/models
	cd cv; python ../train_model.py train/tfidf train/target models

