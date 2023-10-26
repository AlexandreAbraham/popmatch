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


def get_best_group_from_autorank_results(result):
    rankdf = result.rankdf

    # Find the best method
    best = rankdf.meanrank.argsort().iloc[0]
    best_median =  rankdf['median'].iloc[best]
    methods = rankdf[rankdf.ci_upper >= best_median].index

    return methods.values