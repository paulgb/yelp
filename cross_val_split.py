
from sys import argv
import pickle

from random import shuffle

SPLITS = 3

def cross_val_split(outfile, *infile):
    ln = None
    for filename in infile:
        ob = pickle.load(file(filename))
        if ln is not None:
            assert ob.shape[0]
        else:
            ln = ob.shape[0]

    st = range(0, ln)
    shuffle(st)
    split_size = ln/SPLITS
    splits = list()
    for i in range(0, SPLITS):
        split_start = split_size * i
        split_end = split_size * (i + 1)
        test_set = st[split_start:split_end]
        training_set = st[0:split_start] + st[split_end:ln+1]
        assert set(training_set).intersection(set(test_set)) == set()
        assert set(training_set).union(set(test_set)) == set(st)
        splits.append((training_set, test_set))
    pickle.dump(splits, file(outfile, 'w'))

if __name__ == '__main__':
    cross_val_split(*argv[1:])

