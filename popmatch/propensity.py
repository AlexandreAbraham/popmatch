from .experiment import dict_router, dict_wrapper

import numpy as np
import pandas as pd
from psmpy import PsmPy
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV


@dict_wrapper('{splitid}_propensity_score', '{splitid}_propensity_logit')
def propensity_logistic_regression(data_X, data_y, splitid_population, input_random_state,
                                   input_calibrated=True, input_clip_score=0.001):
    
    n_0 = (splitid_population == 0).sum()
    n_1 = (splitid_population == 1).sum()
    n = n_0 + n_1
    sample_weight = None
    if input_calibrated:
        sample_weight = splitid_population.copy()
        sample_weight[splitid_population == 0] = n_1 / n
        sample_weight[splitid_population == 1] = n_0 / n
    clf = LogisticRegression(random_state=input_random_state)
    clf.fit(data_X, data_y, sample_weight=sample_weight)
    propensity_score = clf.predict_proba(data_X)[:, 1]

    if input_clip_score is not None:
        propensity_score = np.clip(propensity_score, input_clip_score, 1 - input_clip_score)
    
    return propensity_score, np.log(propensity_score / (1 - propensity_score))


@dict_wrapper('{splitid}_propensity_score', '{splitid}_propensity_logit')
def propensity_psmpy(data_X, data_y, splitid_population, input_random_state,
                                   input_calibrated=True, input_clip_score=0.001):
    
    df = pd.DataFrame(data_X)
    df['groups'] = splitid_population
    df['index'] = np.arange(data_X.shape[0])

    psm = PsmPy(df, treatment='groups', indx='index', exclude = [], seed=input_random_state)
    psm.logistic_ps(balance=input_calibrated)

    return psm.predicted_data['propensity_score'], psm.predicted_data['propensity_logit']


@dict_wrapper('{splitid}_propensity_score', '{splitid}_propensity_logit')
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
    elif input_propensity_model == 'random_forest':
        return propensity_random_forest
    elif input_propensity_model == 'psmpy':
        return propensity_psmpy