from .utils import cumulate
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.optimize import linear_sum_assignment



def propensity_match(df, population, index_col, categorical, ordinal, continuous, n_match=1, n_limit=10000, verbose=1):
    # Categories must be ordered by importance

    # The principle of the algorithm is the following:
    # - We split by cateogrical variables
    # - If the size of the group is reasonable, we perform bipartite match
    # - Else, we continue to break it down using ordinal variables
    # - If we run out of categorical and we are still not good, then do nn matching.

    groups_to_process = []
    
    for _, category_df in df.groupby(categorical):
        counts = category_df[population].value_counts()
        if counts.min() == 0:
            # One population is missing, skip
            continue

        # If the group is small enough, we keep it
        if counts.max() <= n_limit or len(ordinal) == 0:
            groups_to_process.append((category_df, 0, len(ordinal) == 0))
            continue

        # Otherwise, we use ordinal features.
        for n_features, features in enumerate(cumulate(ordinal)):
            max_group_size = category_df.groupby(features)[population].count().max()
            is_last = (len(features) == len(ordinal))
            if max_group_size <= n_limit or is_last:
                groups_to_process.extend([(g[1], n_features + 1, max_group_size > n_limit) for g in category_df.groupby(features)])
                break
        
    distances = - np.ones(df.shape[0])
    indices = []
    groups = []

    for sub_df, ordinal_idx, use_bipartite in groups_to_process:

        pop = sub_df[population]
        sub_df_0, sub_df_1 = sub_df[pop == 0], sub_df[pop == 1]

        if min(sub_df_0.shape[0], sub_df_1.shape[0]) == 0:
            continue

        if n_match > 1:
            sub_df_0 = pd.concat([sub_df_0] * n_match)

        for i in range(1, 10):
            try:
                ps_dis = NearestNeighbors(n_neighbors=5 * i)
                ps_dis.fit(sub_df_0[['Propensity']])
                ps_dis = ps_dis.radius_neighbors_graph(sub_df_1[['Propensity']], mode='distance')
                fe_dis = NearestNeighbors(n_neighbors=5 * i)
                feats = ordinal[ordinal_idx:] + continuous
                fe_dis.fit(sub_df_0[feats])
                fe_dis = fe_dis.radius_neighbors_graph(sub_df_1[feats], mode='distance')
                ps_dis, fe_dis = ps_dis.T, fe_dis.T
                dis = 1. * ps_dis + fe_dis

                if dis.getnnz() == np.multiply(*dis.shape):
                    pop_0_idx, pop_1_idx = linear_sum_assignment(dis.todense())
                    print('lol', pop_0_idx, pop_1_idx)
                else:
                    pop_0_idx, pop_1_idx = min_weight_full_bipartite_matching(dis)
                    print(pop_0_idx, pop_1_idx)
                sub_df_0 = sub_df_0.iloc[pop_0_idx]
                sub_df_1 = sub_df_1.iloc[pop_1_idx]
                distances[sub_df_0.index[pop_0_idx]] = dis[pop_0_idx, pop_1_idx].A1
                distances[sub_df_1.index[pop_1_idx]] = dis[pop_0_idx, pop_1_idx].A1

                uniques = np.unique(sub_df_0.index)
                indices.append(uniques)
                groups.append(np.zeros(uniques.shape[0]))

                indices.append(sub_df_1.index.values)
                print(sub_df_1.index.values.shape, sub_df_1.index.values)
                print('-')
                groups.append(np.ones(sub_df_1.shape[0]))
                break

            except Exception as e:
                print(e)
                pass

    return np.hstack(indices), np.hstack(groups)