#!/usr/bin/env python

from distutils import setup

setup(
        name='Yelp Review Predictor',
        version='0.1',
        install_requires = [
            'Theano >= 0.6.0rc3',
            'joblib >= 0.7.0d',
            'nltk >= 2.0.4',
            'pandas >= 0.10.1',
            'sklearn-pandas >= 0.0.2',
            'scipy >= 0.11.0',
            'numpy >= 1.6.1'
        ],
        dependency_links = [
            'https://github.com/lmjohns3/py-cli/archive/master.tar.gz#egg=py-cli',
            'https://github.com/lmjohns3/theano-nets/archive/master.tar.gz#egg=theano-nets'
        ]
)

