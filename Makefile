
training_set_json :
	unzip source_data/yelp_training_set.zip
	mv yelp_training_set training_set_json

training_set_csv : training_set_json convert.py
	cd training_set_json; python ../convert.py
	mkdir -p training_set_csv
	mv training_set_json/*.csv training_set_csv/

stemmed/training_set_reviews.dat : stem.py training_set_csv
	mkdir -p stemmed
	python stem.py training_set_csv/yelp_training_set_review.csv stemmed/training_set_reviews.dat

tfidf/vocab.dat : training_set_csv tfidf_vocab.py stemmed/training_set_reviews.dat
	mkdir -p tfidf
	python tfidf_vocab.py stemmed/training_set_reviews.dat tfidf/vocab.dat

tfidf/training_set.dat : tfidf/vocab.dat tfidf.py
	python tfidf.py tfidf/vocab.dat stemmed/training_set_reviews.dat tfidf/training_set.dat

training_set_preproc : training_set_csv preprocess.py
	mkdir -p training_set_preproc
	python preprocess.py training_set_csv/yelp_training_set_review.csv training_set_preproc/reviews.csv


