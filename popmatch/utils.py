from itertools import islice
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler



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


def get_best_group_from_dbscan(df, top_col, threshold_mask=None):
    values = df[top_col].copy()
    if threshold_mask is not None:
        values[~threshold_mask] += np.inf
    i_best = values.abs().argmin()
    X = df.values
    X = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=0.3, min_samples=2).fit(X)
    labels = db.labels_
    if labels[i_best] == -1:
        return np.array([i_best])
    return np.where(labels == labels[i_best])[0]