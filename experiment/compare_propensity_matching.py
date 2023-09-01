import sys
sys.path.append('../')
import pandas as pd
from psmpy import PsmPy
import numpy as np
from statsmodels.stats.weightstats import ttest_ind
from popmatch.data import generate_dataset

from statsmodels.stats.meta_analysis import effectsize_smd
from formulaic import ModelSpec, Formula
from formulaic.parser.types.factor import Factor
from popmatch.match import propensity_match_cv
from popmatch.splitter import GeneralPurposeClustering
from sklearn.preprocessing import StandardScaler
from popmatch.utils import compute_smd


# dataset = generate_dataset(10000, 1000, 4, [2, 4, 6, 8], 4, 4, random_state=12)
df = pd.read_csv('../datasets/heart.csv')

#X_ = Formula(
#    'scale(age) + C(sex) + C(cp) + scale(trestbps) + C(chol) + C(fbs)' \
#    ' + scale(restecg) + C(thalach) + scale(exang) + C(oldpeak) + C(slope)' \
#    ' + C(ca) + C(thal)').get_model_matrix(dataset['X'])
# y = dataset['y']
# specs = ModelSpec.from_spec(X)

target = 'target'
continuous = ['age', 'trestbps', 'chol', 'thalach',  'oldpeak']
ordinal = ['restecg', 'ca']
categorical = ['sex', 'cp', 'fbs',  'exang', 'slope', 'thal']

continuous_std = []
for c in continuous:
    df[c + '_std'] = StandardScaler().fit_transform(df[[c]])
    continuous_std.append(c + '_std')
import os


class MyLoss():

    def __init__(self, target, continuous, categorical):
        self.target = target
        self.continuous = continuous
        self.categorical = categorical

    def __call__(self, df, cluster_ids):
        y_0 = df[self.target][cluster_ids == 0].mean()
        y_1 = df[self.target][cluster_ids == 1].mean()
        # Let's target a difference of 0.2
        dy = (0.2 - (y_1 - y_0)) ** 2

        smds = compute_smd(df, cluster_ids, self.continuous, self.categorical)
        smd = smds.smd.mean()
        
        return dy - 10 * smd


if not os.path.exists('cluster_id.csv'):
    gps = GeneralPurposeClustering([0.3, 0.7], MyLoss(target, continuous_std, categorical + ordinal), verbose=1)
    gps.fit(df)
    np.savetxt('cluster_id.csv', gps.cluster_id_)

groups = np.loadtxt('cluster_id.csv')
df['groups'] = groups
df['index'] = np.arange(df.shape[0])

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
my_groups, matchmap, _, _ = propensity_match_cv(df, 'groups', 'Propensity', categorical, ordinal, continuous, n_match=1, verbose=1, feature_weight=1.0)
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
