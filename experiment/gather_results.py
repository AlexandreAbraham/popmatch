import numpy as np
import pandas as pd
import os
from autorank import autorank
import sys
from scipy.stats import kendalltau

sys.path.append('../')
from popmatch.utils import get_best_group_from_autorank_results


datasets = [
    'groupon',
    'nhanes',
    'horse',
    'synthetic_7_0',
    'synthetic_7_1',
    'synthetic_7_2',
    'synthetic_7_3',
    'synthetic_7_4',
    'synthetic_7_5',
    'synthetic_7_6',
    'synthetic_7_7',
    'synthetic_7_8',
    'synthetic_7_9',
    'synthetic_7_10',
]

results = []
artificials = []

for dataset in datasets:
    filename = f"{dataset}/results.parquet"
    ar_filename = f"{dataset}/artificial.parquet"
    if not os.path.exists(filename) or not os.path.exists(ar_filename):
        print(f'{dataset} not loaded')
        continue
    result = pd.read_parquet(filename)
    result['dataset'] = dataset
    results.append(result)
    artificial = pd.read_parquet(ar_filename)
    artificial['dataset'] = dataset
    artificials.append(artificial)

results = pd.concat(results)
artificials = pd.concat(artificials)
artificials['split'] = artificials['split_id'].str.extract(r'split(\d+)$')


def _add_bounds(df, values, errors, prefix):
    mins = None
    maxs = None
    gaps = None
    avg_error = None
    if values.shape[0] > 0:     
        mins = values.min()
        maxs = values.max()
        gaps = np.abs(maxs - mins)
        if errors is not None:
            avg_error = np.mean(np.abs(errors))
    df[prefix + 'target_min'] = mins
    df[prefix + 'target_max'] = maxs
    df[prefix + 'target_gap'] = gaps
    df[prefix + 'avg_error'] = avg_error


class Bounds:
    def __init__(self, artificial_df, value_col, error_col,
                 threshold_col, top_col,
                 top_rank='auto', threshold=0.1):
        self.artificial_df = artificial_df.set_index(['dataset', 'split', 'matching']).unstack(level=-1).target.abs()
        self.value_col = value_col
        self.error_col = error_col
        self.threshold_col = threshold_col
        self.threshold = threshold
        self.top_col = top_col
        self.top_rank = top_rank

    def __call__(self, df):

        artificial_df = self.artificial_df.loc[df.group_dataset.iloc[0], :]

        # SMD
        filtered_df = df[df[self.threshold_col] < self.threshold]
        _add_bounds(df, filtered_df[self.value_col].values, filtered_df[self.error_col].values, self.threshold_col)

        # A2A
        ar_result = autorank(artificial_df[df.matching.unique()], alpha=0.05, force_mode='nonparametric')
        ar_best_methods = get_best_group_from_autorank_results(ar_result)
        filtered_df = df[df.matching.isin(ar_best_methods)]
        _add_bounds(df, filtered_df[self.value_col].values, filtered_df[self.error_col].values, self.top_col)

        # A2A-SMD
        filtered_df =  filtered_df[filtered_df[self.threshold_col] < self.threshold]
        _add_bounds(df, filtered_df[self.value_col].values, filtered_df[self.error_col].values, f'{self.top_col}-{self.threshold_col}')

        # SMD-A2A
        filtered_df = df[df[self.threshold_col] < self.threshold]
        ar_result = autorank(artificial_df[filtered_df.matching.unique()], alpha=0.05, force_mode='nonparametric')
        ar_best_methods = get_best_group_from_autorank_results(ar_result)
        filtered_df = filtered_df[filtered_df.matching.isin(ar_best_methods)]
        _add_bounds(df, filtered_df[self.value_col].values, filtered_df[self.error_col].values, f'{self.threshold_col}-{self.top_col}')

        return df

results = results[~results['method'].str.contains('logit')]
artificials = artificials[~artificials['method'].str.contains('logit')]

results['absdiff'] = results['diff'].abs()
results['group_dataset'] = results['dataset']  # Ugly hack because I need to know the dataset in the grouping...
agg = Bounds(artificials, 'target', 'ate', 'smd', 'absdiff')
results = results.groupby('dataset').apply(agg).reset_index(drop=True)


results = results.fillna(-99999).groupby('dataset').mean()
results = results[['smdtarget_gap', 'absdifftarget_gap', 'smd-absdifftarget_gap', 'absdiff-smdtarget_gap',
                   'smdavg_error', 'absdiffavg_error', 'smd-absdiffavg_error', 'absdiff-smdavg_error']]

#print(results)
print(results.to_latex(float_format="%.3f"))