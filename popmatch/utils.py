from itertools import islice


def cumulate(l):
    r = []
    for i in l:
        r.append(i)
        yield r


def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch