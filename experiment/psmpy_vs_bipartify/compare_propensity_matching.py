import sys
sys.path.append('../../')
import pandas as pd
from psmpy import PsmPy
import numpy as np
from statsmodels.stats.weightstats import ttest_ind
from popmatch.data import load_data, standardize_continuous_features

from statsmodels.stats.meta_analysis import effectsize_smd
from formulaic import ModelSpec, Formula
from formulaic.parser.types.factor import Factor
from popmatch.match import bipartify, split_populations_with_error, psmpy_match, matchit_match
from popmatch.evaluation import compute_smd, compute_target_mean_difference
from popmatch.preprocess import preprocess
from popmatch.propensity import propensity_score
from popmatch.experiment import dict_cache
from popmatch.plot import plot_smds
from scipy.stats import ttest_rel
import itertools
import tqdm


experiment = {
    'input': {
        'dataset': 'heart',
        'propensity_model': 'logistic_regression',
        #'propensity_model': 'random_forest',
        #'propensity_model': 'psmpy',
        'propensity_transform': 'identity',
        # 'propensity_transform': 'logit',
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

transforms = ['identity', 'logit']
models = ['logistic_regression', 'random_forest', 'psmpy']
for transform, model in itertools.product(transforms, models):
    split_ids = []
    targets = []
    for seed in tqdm.tqdm(list(range(0, 100))):
        split_id = 'split' + str(seed)
        split_ids.append(split_id)
        for not_already in dict_cache(experiment, split_id):
            split_populations_with_error(experiment, input_random_state=seed, splitid=split_id)
        matchit_match(experiment, splitid=split_id)
        propensity_score(experiment, input_random_state=seed, splitid=split_id,
                         input_propensity_model=model, input_propensity_transform=transform)
        bipartify(experiment, splitid=split_id, input_random_state=seed, n_match=1, feature_weight=0.1, verbose=1)
        psmpy_match(experiment, splitid=split_id, input_random_state=seed)

        bipartify_smd = compute_smd(experiment, splitid=split_id, matching='bipartify').smd.mean()
        bipartify_n0, bipartify_n1, bipartify_target_diff = compute_target_mean_difference(experiment, splitid=split_id, matching='bipartify')
        targets.append({'n0': bipartify_n0, 'n1': bipartify_n1, 'target': bipartify_target_diff, 'matching': 'bipartify', 'split_id': split_id})

        psmpy_smd = compute_smd(experiment, splitid=split_id, matching='psmpy').smd.mean()
        psmpy_n0, psmpy_n1, psmpy_target_diff = compute_target_mean_difference(experiment, splitid=split_id, matching='psmpy')
        targets.append({'n0': psmpy_n0, 'n1': psmpy_n1, 'target': psmpy_target_diff, 'matching': 'psmpy', 'split_id': split_id})
        
    # compute_smd_baseline(experiment)

    print(f'Results for {transform} {model}')
    plot_smds(experiment, split_ids, ['bipartify', 'psmpy'], f'smds_{transform}_{model}.png')
    targets = pd.DataFrame.from_records(targets).sort_values(by='split_id')
    targets_b = targets[targets.matching == 'bipartify'].target.values
    targets_p = targets[targets.matching == 'psmpy'].target.values
    print(targets.groupby('matching')[['n0', 'n1', 'target']].mean())
    print('Bipartify wins {} percent of the times'.format((targets_b < targets_p).mean() * 100))
    print(targets.groupby('matching').target.describe())
    print(ttest_rel(targets_p, targets_b))
    print('\n\n')

