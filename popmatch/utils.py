import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from formulaic import Formula, ModelSpec
import scipy.stats as stats
import numpy as np
import math


class FormulaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, formula):
        self.formula = Formula(formula)
    
    def fit(self, X, y=None):
        self.model_spec_ = ModelSpec.from_spec(self.formula.get_design_matrix(X))
        return self
    
    def transform(self, X):
        return self.model_spec_.get_design_matrix(X)
    

def cumulate(l):
    r = []
    for i in l:
        r.append(i)
        yield r


def smd(df, formula):
    f = Formula(formula)
    dm = f.get_matrix_model(df).values
    ms = ModelSpec.from_spec(dm)
    group = str(f.lhs)
    assert(np.unique(df[group]) == np.array([0, 1]))

    smds = []
    for feature in ms.rhs.term_indices:
        if feature == 1:
            continue

        indices = ms.rhs.term_indices[feature]

        if len(indices) == 1:
            # Continuous
            v = df[:, indices[0]]
            v0, v1 = v[df[group] == 0], v[df[group] == 1]
            rn0, rn1 = np.shape[0] - 1, n1.shape[0] - 1

            denominator = math.sqrt((rn1 * v1.std() ** 2 + rn0 * v0.std() ** 2) / (rn1 + rn0))
            denominator = max(1e-6, denominator)
            smd = abs((v1.mean() - v0.mean()) / denominator)

        else:
            # Categorical
            v = df[:, indices].values
            v0 = v[df[group] == 0]
            v1 = v[df[group] == 1]

            contingency_table = np.array([v0.sum(axis=0), v1.sum(axis=0)])
            x2 = stats.chi2_contingency(contingency_table, correction=False)[0]
  
            smd = np.sqrt((x2 / v.shape[0]) / (len(indices)))
        smds.append((feature, smd))
    return smds