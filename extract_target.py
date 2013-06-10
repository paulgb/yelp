
from sys import argv
import pickle

import pandas as pd
from numpy import log

def extract_target(infile, outfile):
    reviews = pd.read_csv(infile, parse_dates=[1], na_values=[], keep_default_na=False)
    pickle.dump(log(reviews.votes_useful + 1), file(outfile, 'w'))

if __name__ == '__main__':
    extract_target(*argv[1:])
