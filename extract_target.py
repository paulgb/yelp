
from sys import argv
import pickle

import pandas as pd

def extract_target(infile, outfile):
    reviews = pd.read_csv(infile, parse_dates=[1], na_values=[], keep_default_na=False)
    pickle.dump(reviews.votes_useful, file(outfile, 'w'))

if __name__ == '__main__':
    extract_target(*argv[1:])
