from .utils import compute_smd
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler



def propensity_match(df, population, propensity,
                     categorical, continuous,
                     n_match=1, feature_weight=0.1, standardize=True, verbose=1):
    """Perform population matching.


    
    
    """


    # Categories must be ordered by importance

    # The principle of the algorithm is the following:
    # - We split by cateogrical variables
    # - If the size of the group is reasonable, we perform bipartite match
    # - Else, we continue to break it down using ordinal variables
    # - If we run out of categorical and we are still not good, then do nn matching.
    
    continuous_features = df[continuous]
    if standardize:
        continuous_features = StandardScaler().fit_transform(continuous_features)
        continuous_features = pd.DataFrame(continuous_features, columns=continuous)

    match_pop_0 = []
    match_pop_1 = []
    match_dists = []

    for _, gdf in df.groupby(categorical):

        pop = gdf[population]
        gdf_0, gdf_1 = gdf[pop == 0], gdf[pop == 1]

        if min(gdf_0.shape[0], gdf_1.shape[0]) == 0:
            continue

        if n_match > 1:
            gdf_0 = pd.concat([gdf_0] * n_match)

        for i in range(1, 10):
            try:
                ps_dis = NearestNeighbors(n_neighbors=5 * i)
                ps_dis.fit(gdf_0[[propensity]])
                ps_dis = ps_dis.radius_neighbors_graph(gdf_1[[propensity]], mode='distance')
                fe_dis = NearestNeighbors(n_neighbors=5 * i)
                fe_dis.fit(continuous_features.loc[gdf_0.index])
                fe_dis = fe_dis.radius_neighbors_graph(continuous_features.loc[gdf_1.index], mode='distance')
                ps_dis, fe_dis = ps_dis.T, fe_dis.T
                dis = ps_dis + feature_weight * fe_dis

                # If all distances are defined, bipartite match stalls. We use hungarian in this case
                if dis.getnnz() == np.multiply(*dis.shape):
                    pop_0_idx, pop_1_idx = linear_sum_assignment(dis.todense())
                else:
                    pop_0_idx, pop_1_idx = min_weight_full_bipartite_matching(dis)
                match_pop_0.append(gdf_0.index[pop_0_idx].values)
                match_pop_1.append(gdf_1.index[pop_1_idx].values)
                match_dists.append(dis[pop_0_idx, pop_1_idx].A1)

                break

            except Exception as e:
                print(e)
                pass

    # We create a group indicator and return it.
    groups = pd.DataFrame(-np.ones(df.shape[0]), index=df.index)
    match_pop_0 = np.hstack(match_pop_0)
    match_pop_1 = np.hstack(match_pop_1)
    match_dists = np.hstack(match_dists)
    groups.loc[match_pop_0] = 0
    groups.loc[match_pop_1] = 1

    matchmap = pd.DataFrame.from_dict({'index_pop_0': match_pop_0, 'index_pop_1': match_pop_1, 'distance': match_dists})

    return groups, matchmap


def propensity_match_cv(df, population, propensity,
                        categorical, ordinal, continuous,
                        n_match=1, feature_weight=0.1, verbose=1):
    
    best_groups = None
    best_matchmap = None
    best_smd = None
    best_idx = None

    # Ordinal features must be ordered by decreasing importance.
    for i in range(len(ordinal) + 1):
        groups, matchmap = propensity_match(df, population, propensity,
                                            categorical + ordinal[:i], continuous + ordinal[i:],
                                            n_match=n_match, feature_weight=feature_weight,
                                            verbose=verbose)
        mask = (groups >= 0).values
        smd = compute_smd(df[mask], groups.values[mask], continuous, categorical + ordinal)
        smd = smd.smd.mean()
        if best_smd is None or smd < best_smd:
            best_smd = smd
            best_groups = groups
            best_matchmap = matchmap
            best_idx = i

    return best_groups, best_matchmap, best_smd, best_idx
