from .experiment import dict_router, dict_wrapper

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV


@dict_wrapper('propensity_score', 'propensity_logit')
def propensity_logistic_regression(data_df, data_X, data_y, data_population,
                                   input_calibrated=True, input_clip_score=0.001):
    
    n_0 = (data_population == 0).sum()
    n_1 = (data_population == 1).sum()
    n = n_0 + n_1
    sample_weight = None
    if input_calibrated:
        sample_weight = data_population.copy()
        sample_weight[data_population == 0] = n_1 / n
        sample_weight[data_population == 1] = n_0 / n
    clf = LogisticRegression()
    clf.fit(data_X, data_y, sample_weight=sample_weight)
    propensity_score = clf.predict_proba(data_X)[:, 1]

    if input_clip_score is not None:
        propensity_score = np.clip(propensity_score, input_clip_score, 1 - input_clip_score)
    
    return propensity_score, np.log(propensity_score / (1 - propensity_score))


@dict_wrapper('propensity_score', 'propensity_logit')
def propensity_random_forest(data_X, data_y,
                             input_calibrated=True, input_clip_score=0.001):
    
    clf = RandomForestClassifier()
    if input_calibrated:
        clf.fit(data_X, data_y)
        clf = CalibratedClassifierCV(base_estimator=clf, method='sigmoid', cv='prefit')

    clf.fit(data_X, data_y)
    propensity_score = clf.predict_proba(data_X)[:, 1]

    if input_clip_score is not None:
        propensity_score = np.clip(propensity_score, input_clip_score, 1 - input_clip_score)
    
    return propensity_score, np.log(propensity_score / (1 - propensity_score))


@dict_router
def propensity_score(input_propensity_model):

    if input_propensity_model == 'logistic_regression':
        return propensity_logistic_regression