
from math import ceil

from random import shuffle

from mem import cache

@cache
def split(table, nsplits):
    splits = list()
    ln = table.shape[0]
    st = range(0, ln)
    shuffle(st)

    split_size = ln/nsplits
    for i in range(0, nsplits):
        split_start = split_size * i
        split_end = split_size * (i + 1)
        test_set = st[split_start:split_end]
        training_set = st[0:split_start] + st[split_end:ln+1]
        splits.append((table.irow(training_set), table.irow(test_set)))

    return splits

@cache
def partition(table, frac):
    ln = table.shape[0]
    st = range(0, ln)
    shuffle(st)

    split_size = int(ceil(ln * (1 - frac)))

    training_set = st[:split_size]
    test_set = st[split_size:]

    return table.irow(training_set), table.irow(test_set)


