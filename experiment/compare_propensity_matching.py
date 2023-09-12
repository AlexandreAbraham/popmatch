import sys
sys.path.append('../')
import pandas as pd
from psmpy import PsmPy
import numpy as np
from statsmodels.stats.weightstats import ttest_ind
from popmatch.data import load_data, standardize_continuous_features

from statsmodels.stats.meta_analysis import effectsize_smd
from formulaic import ModelSpec, Formula
from formulaic.parser.types.factor import Factor
from popmatch.match import bipartify, split_populations_with_error
from popmatch.evaluation import compute_smd
from popmatch.preprocess import preprocess
from popmatch.propensity import propensity_score


experiment = {
    'input': {
        'dataset': 'heart',
        'propensity_model': 'logistic_regression',
        'clip_score': 0.05,
        'calibrated': True,
        'random_state': 12,
        'simulated_split_population_ratio': [0.3, 0.7],
        'simulated_split_target_difference': 0.2,
        'simulated_split_smd_weight': 10,
    },
}

load_data(experiment)
standardize_continuous_features(experiment)
preprocess(experiment)
for seed in range(0, 1):
    split_populations_with_error(experiment, data_random_state=seed)
    propensity_score(experiment, data_random_state=seed)
    bipartify(experiment, matching='seed0', data_random_state=seed, n_match=1, feature_weight=0.1, verbose=1)

    our_smd = compute_smd(experiment, matching='seed0').smd.mean()

print('SMD', our_smd)
# compute_smd_baseline(experiment)



if False:
    # Forget about psmpy for now
    experiment['data']['df']['index'] = np.arange(experiment['data']['df'].shape[0])

    psm = PsmPy(df[categorical + continuous_std + ['groups', 'index']], treatment='groups', indx='index', exclude = [])
    psm.logistic_ps(balance=True)
    df['Propensity'] = psm.predicted_data['propensity_score']
    print(df['Propensity'])

    m = psm.knn_matched(matcher='propensity_logit', replacement=False, caliper=1.7, drop_unmatched=True)

    # Establish reference SMD
    all_smds = []
    for i in range(100):
        random_groups = np.random.choice(2, size=df.shape[0], replace=True)
        smd = compute_smd(df, random_groups, continuous_std, categorical + ordinal)
        all_smds.append(smd.smd.mean())
    all_smds = np.array(all_smds)

    print(f'SMD reference, {all_smds.mean()} ({all_smds.std()})')
    print(f'SMD original', compute_smd(df, groups, continuous_std, categorical + ordinal).smd.mean())
    matched_idx = np.hstack([psm.matched_ids["index"].values, psm.matched_ids["matched_ID"].values])
    matched_groups = np.hstack([np.zeros(psm.matched_ids["index"].shape[0]), np.ones(psm.matched_ids["matched_ID"].shape[0])])
    matched_df = df.iloc[matched_idx].copy()
    print(f'SMD matched psmpy', compute_smd(matched_df, matched_groups, continuous_std, categorical + ordinal).smd.mean())


print('Prop')
#my_groups, matchmap, _, _ = bipartify_cv(df, 'groups', 'Propensity', categorical, ordinal, continuous, n_match=1, verbose=1, feature_weight=1.0)
my_groups = my_groups.values[:, 0]
mask = (my_groups >= 0)
print(f'SMD matched ours', compute_smd(df[mask], my_groups[mask], continuous_std, categorical + ordinal).smd.mean())

print('Size psmpy', (matched_groups == 0).sum(), (matched_groups == 1).sum())
print('Size ours', (my_groups == 0).sum(), (my_groups == 1).sum())


print(f'Target mean group 0 {df[target][groups == 0].mean()}')
print(f'Target mean group 1 {df[target][groups == 1].mean()}')

print(f'Target mean psmpy group 0 {df[target][psm.matched_ids["index"]].mean()}')
print(f'Target mean psmpy group 1 {df[target][psm.matched_ids["matched_ID"]].mean()}')

print(f'Target mean ours group 0 {df[target][my_groups == 0].mean()}')
print(f'Target mean ours group 1 {df[target][my_groups == 1].mean()}')

print('p-value matched', ttest_ind(df[target][psm.matched_ids['index']], df[target][psm.matched_ids['matched_ID']])[1])
