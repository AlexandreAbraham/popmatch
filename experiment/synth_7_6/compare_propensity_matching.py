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
from popmatch.match import bipartify, split_populations_with_error, split_stats, psmpy_match, matchit_match, load_biggest_population, load_real_problem
from popmatch.evaluation import compute_smd, compute_target_mean_difference, compute_simulation_params, compute_synth_metrics
from popmatch.preprocess import preprocess
from popmatch.propensity import propensity_score
from popmatch.experiment import dict_cache
from popmatch.plot import plot_smds
from scipy.stats import ttest_rel
import itertools
import tqdm


experiment = {
    'input': {
        'dataset': 'synthetic_7_6',
        #'propensity_model': 'logistic_regression',
        #'propensity_model': 'random_forest',
        #'propensity_model': 'psmpy',
        'propensity_transform': 'identity',
        # 'propensity_transform': 'logit',
        'clip_score': 0.05,
        'calibrated': True,
        'random_state': 12,
        #'simulated_split_population_ratio': [0.3, 0.7],
        #'simulated_split_target_difference': 0.2,
        #'simulated_split_smd_weight': 10,
    },
}

for not_already in dict_cache(experiment, 'data', cache_path='./data_cache'):
    load_data(experiment)
standardize_continuous_features(experiment)
preprocess(experiment)
compute_simulation_params(experiment)


transforms = ['identity', 'logit']
models = ['logistic-regression', 'random-forest', 'psmpy']
distances = ['glm', 'gam',  'elasticnet', 'rpart',
                'cbps', 'bart',
                 ]  # 'gbm', 'randomforest', 'nnet', 'scaled_euclidean','robust_mahalanobis',
methods = ['nearest', 'optimal',# 'genetic',
            ]  # 'cardinality',, 'full', 'cem', 'exact', 'subclass'


load_biggest_population(experiment)
targets = []
for seed in tqdm.tqdm(list(range(0, 50))):
    split_id = 'split' + str(seed)
    for not_already in dict_cache(experiment, split_id, cache_path='./split_cache'):
        split_populations_with_error(experiment, input_random_state=seed, splitid=split_id)

    print(split_id)
    split_stats(experiment, splitid=split_id)

    msplit_id = 'matchit' + split_id
    for not_already in dict_cache(experiment, msplit_id, cache_path='./matchit_cache'):
        for distance, method in itertools.product(distances, methods):
            matchit_match(experiment, splitid=split_id, distance=distance, method=method)
            matchit_smd = compute_smd(experiment, splitid=msplit_id, matching=f'{distance}_{method}').smd.mean()
            matchit_n0, matchit_n1, matchit_target_diff, matchit_ind_pvalue, matchit_rel_pvalue = compute_target_mean_difference(
                experiment, splitid=msplit_id, matching=f'{distance}_{method}')
            targets.append({'n0': matchit_n0, 'n1': matchit_n1, 'target': matchit_target_diff, 'split_id': msplit_id})
            print(f'{distance}_{method}', matchit_smd, matchit_n0, matchit_n1, matchit_target_diff, matchit_ind_pvalue, matchit_rel_pvalue)
    res = experiment[msplit_id]
    for distance, method in itertools.product(distances, methods):
        m = f'{distance}_{method}'
        targets.append({'smd': res[f'{m}_smds'].smd.mean(), 'n0': res[f'{m}_n0'], 'n1': res[f'{m}_n1'], 'target': res[f'{m}_target_diff'], 'matching': m, 'split_id': msplit_id})
            
    for transform, model in itertools.product(transforms, models):
        psplit_id = f'python{transform}{model}{split_id}'
        for not_already in dict_cache(experiment, psplit_id, cache_path='./python_cache'):
            propensity_score(experiment, input_random_state=seed, splitid=split_id, output=psplit_id,
                            input_propensity_model=model, input_propensity_transform=transform)
            bipartify(experiment, splitid=psplit_id, input_random_state=seed, n_match=1, feature_weight=0.1, verbose=1)
            psmpy_match(experiment, splitid=psplit_id, input_random_state=seed)

            bipartify_smd = compute_smd(experiment, splitid=psplit_id, matching='bipartify').smd.mean()
            bipartify_n0, bipartify_n1, bipartify_target_diff, bipartify_ind_pvalue, bipartify_rel_pvalue = \
                compute_target_mean_difference(experiment, splitid=psplit_id, matching='bipartify')
            print(f'bipartify', transform, model, bipartify_smd, bipartify_n0, bipartify_n1,
                   bipartify_target_diff, bipartify_ind_pvalue, bipartify_rel_pvalue)

            psmpy_smd = compute_smd(experiment, splitid=psplit_id, matching='psmpy').smd.mean()
            psmpy_n0, psmpy_n1, psmpy_target_diff, psmpy_ind_pvalue, psmpy_rel_pvalue = \
                compute_target_mean_difference(experiment, splitid=psplit_id, matching='psmpy')
            print(f'psmpy', transform, model, psmpy_smd, psmpy_n0, psmpy_n1, psmpy_target_diff,
                   psmpy_ind_pvalue, psmpy_rel_pvalue)
        res = experiment[psplit_id]
        m = 'bipartify'
        targets.append({'smd': res[f'{m}_smds'].smd.mean(), 'n0': res[f'{m}_n0'], 'n1': res[f'{m}_n1'], 'target': res[f'{m}_target_diff'], 'matching': m, 'split_id': psplit_id})
        m = 'psmpy'
        targets.append({'smd': res[f'{m}_smds'].smd.mean(), 'n0': res[f'{m}_n0'], 'n1': res[f'{m}_n1'], 'target': res[f'{m}_target_diff'], 'matching': m, 'split_id': psplit_id})    


real_targets = []
for _ in range(1):

    split_id = 'realxp'
    load_real_problem(experiment, splitid=split_id)
    print(split_id)

    msplit_id = 'matchit' + split_id
    for not_already in dict_cache(experiment, msplit_id, cache_path='./realxp_cache'):
        for distance, method in itertools.product(distances, methods):
            matchit_match(experiment, splitid=split_id, distance=distance, method=method)
            matchit_smd = compute_smd(experiment, splitid=msplit_id, matching=f'{distance}_{method}').smd.mean()
            matchit_n0, matchit_n1, matchit_target_diff, matchit_ind_pvalue, matchit_rel_pvalue = compute_target_mean_difference(
                experiment, splitid=msplit_id, matching=f'{distance}_{method}')
            matchit_ate_diff, matchit_outcome_diff, matchit_ite_diff = compute_synth_metrics(experiment, splitid=msplit_id, matching=f'{distance}_{method}')
            print(f'{distance}_{method}', matchit_smd, matchit_n0, matchit_n1, matchit_target_diff, matchit_ind_pvalue, matchit_rel_pvalue)
    res = experiment[msplit_id]
    for distance, method in itertools.product(distances, methods):
        m = f'{distance}_{method}'
        real_targets.append({'smd': res[f'{m}_smds'].smd.mean(), 'n0': res[f'{m}_n0'], 'n1': res[f'{m}_n1'],
                             'target': res[f'{m}_target_diff'], 'ate': res[f'{m}_ate_diff'],
                             'ite': res[f'{m}_ite_diff'], 'outcome': res[f'{m}_outcome_diff'],
                             'matching': m, 'split_id': msplit_id})
     
    for transform, model in itertools.product(transforms, models):
        psplit_id = f'python{transform}{model}{split_id}'
        for not_already in dict_cache(experiment, psplit_id, cache_path='./realxp_cache'):
            propensity_score(experiment, input_random_state=seed, splitid=split_id, output=psplit_id,
                            input_propensity_model=model, input_propensity_transform=transform)
            bipartify(experiment, splitid=psplit_id, input_random_state=seed, n_match=1, feature_weight=0.1, verbose=1)
            psmpy_match(experiment, splitid=psplit_id, input_random_state=seed)

            bipartify_smd = compute_smd(experiment, splitid=psplit_id, matching='bipartify').smd.mean()
            bipartify_n0, bipartify_n1, bipartify_target_diff, bipartify_ind_pvalue, bipartify_rel_pvalue = \
                compute_target_mean_difference(experiment, splitid=psplit_id, matching='bipartify')
            bipartify_ate_diff, bipartify_outcome_diff, bipartify_ite_diff = compute_synth_metrics(experiment, splitid=psplit_id, matching='bipartify')

            # targets.append({'n0': bipartify_n0, 'n1': bipartify_n1, 'target': bipartify_target_diff, 'matching': 'bipartify', 'split_id': split_id})
            print(f'bipartify', transform, model, bipartify_smd, bipartify_n0, bipartify_n1,
                   bipartify_target_diff, bipartify_ind_pvalue, bipartify_rel_pvalue)

            psmpy_smd = compute_smd(experiment, splitid=psplit_id, matching='psmpy').smd.mean()
            psmpy_n0, psmpy_n1, psmpy_target_diff, psmpy_ind_pvalue, psmpy_rel_pvalue = \
                compute_target_mean_difference(experiment, splitid=psplit_id, matching='psmpy')
            psmpy_ate_diff, psmpy_outcome_diff, psmpy_ite_diff = compute_synth_metrics(experiment, splitid=psplit_id, matching='psmpy')
            # targets.append({'n0': psmpy_n0, 'n1': psmpy_n1, 'target': psmpy_target_diff, 'matching': 'psmpy', 'split_id': split_id})
            print(f'psmpy', transform, model, psmpy_smd, psmpy_n0, psmpy_n1, psmpy_target_diff,
                   psmpy_ind_pvalue, psmpy_rel_pvalue)
        
        res = experiment[psplit_id]
        m = 'bipartify'
        real_targets.append({'smd': res[f'{m}_smds'].smd.mean(), 'n0': res[f'{m}_n0'], 'n1': res[f'{m}_n1'],
                            'target': res[f'{m}_target_diff'], 'ate': res[f'{m}_ate_diff'],
                             'ite': res[f'{m}_ite_diff'], 'outcome': res[f'{m}_outcome_diff'],
                            'matching': m, 'split_id': psplit_id})
        m = 'psmpy'
        real_targets.append({'smd': res[f'{m}_smds'].smd.mean(), 'n0': res[f'{m}_n0'], 'n1': res[f'{m}_n1'],
                            'target': res[f'{m}_target_diff'], 'ate': res[f'{m}_ate_diff'],
                             'ite': res[f'{m}_ite_diff'], 'outcome': res[f'{m}_outcome_diff'],
                             'matching': m, 'split_id': psplit_id})    


targets = pd.DataFrame.from_records(targets)
targets['method'] = targets.split_id.str.split('split').str[0] + targets['matching']

rtargets = pd.DataFrame.from_records(real_targets)
rtargets['method'] = rtargets.split_id.str.split('real').str[0] + rtargets['matching']

a = targets.groupby('method')[['target']].mean().rename(columns={'target': 'diff'}).reset_index()
b = rtargets.groupby('method')[['target', 'ate', 'ite', 'outcome', 'smd']].mean().reset_index()
c = pd.concat([a, b], axis=1)


from matplotlib import pyplot as plt

plt.scatter(c['diff'], c['smd'])
plt.xlabel('diff')
plt.ylabel('smd')
plt.savefig('diff_vs_smd.png')
plt.close()

plt.scatter(c['diff'], c['target'])
plt.xlabel('diff')
plt.ylabel('target')
plt.savefig('diff_vs_target.png')
plt.close()

plt.scatter(c['smd'], c['target'])
plt.xlabel('smd')
plt.ylabel('target')
plt.savefig('smd_vs_target.png')
plt.close()

plt.scatter(c['diff'], c['smd'])
plt.xlabel('diff')
plt.ylabel('smd')
for i, txt in enumerate(c['target']):
    plt.gca().annotate('{:.2}'.format(txt), (c.iloc[i]['diff'], c.iloc[i]['smd']), fontsize=10)
plt.savefig('diff_vs_smd_vs_target.png')
plt.close()

plt.scatter(c['diff'], c['smd'])
plt.xlabel('diff')
plt.ylabel('smd')
for i, txt in enumerate(c['ate']):
    plt.gca().annotate('{:.2}'.format(txt), (c.iloc[i]['diff'], c.iloc[i]['smd']), fontsize=10)
plt.savefig('diff_vs_smd_vs_ate.png')
plt.close()

plt.scatter(c['diff'], c['smd'])
plt.xlabel('diff')
plt.ylabel('smd')
for i, txt in enumerate(c['ite']):
    plt.gca().annotate('{:.2}'.format(txt), (c.iloc[i]['diff'], c.iloc[i]['smd']), fontsize=10)
plt.savefig('diff_vs_smd_vs_ite.png')
plt.close()

plt.scatter(c['diff'], c['smd'])
plt.xlabel('diff')
plt.ylabel('smd')
for i, txt in enumerate(c['outcome']):
    plt.gca().annotate('{:.2}'.format(txt), (c.iloc[i]['diff'], c.iloc[i]['smd']), fontsize=10)
plt.savefig('diff_vs_smd_vs_outcome.png')
plt.close()

plt.scatter(c['diff'], c['smd'])
plt.xlabel('diff')
plt.ylabel('smd')
for i, txt in enumerate(c['target']):
    plt.gca().annotate(c.iloc[i]['method'], (c.iloc[i]['diff'], c.iloc[i]['smd']), fontsize=10)
plt.savefig('diff_vs_smd_vs_method.png')
plt.close()





    # compute_smd_baseline(experiment)

    # print(f'Results for {transform} {model}')
    # plot_smds(experiment, split_ids, ['bipartify', 'psmpy', 'matchit'], f'smds_{transform}_{model}.png')
    # targets = pd.DataFrame.from_records(targets).sort_values(by='split_id')
    # targets_b = targets[targets.matching == 'bipartify'].target.values
    # targets_p = targets[targets.matching == 'psmpy'].target.values
    # targets_m = targets[targets.matching == 'matchit'].target.values

    # print(targets.groupby('matching')[['n0', 'n1', 'target']].mean())
    # print('Bipartify wins {} percent of the times'.format((targets_b < targets_p).mean() * 100))
    # print(targets.groupby('matching').target.describe())
    # print(ttest_rel(targets_p, targets_b))
    #print('\n\n')

