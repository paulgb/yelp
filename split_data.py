
from sys import argv
import pickle

def split_data(splitfile, infile, outfile):
    splits = pickle.load(file(splitfile))
    outsets = list()
    dataset = pickle.load(file(infile))
    for i, (training_split, test_split) in enumerate(splits):
        if hasattr(dataset, 'irow'):
            training_set = dataset.irow(training_split)
            test_set = dataset.irow(test_split)
        else:
            training_set = dataset[training_split]
            test_set = dataset[test_split]
        #training_set = training_set.toarray()
        #test_set = test_set.toarray()
        pickle.dump(training_set, file('train/%s%s.dat' % (outfile, i), 'w'))
        pickle.dump(test_set, file('test/%s%s.dat' % (outfile, i), 'w'))

if __name__ == '__main__':
    split_data(*argv[1:])

