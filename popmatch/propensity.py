from formulaic import Formula
from sklearn.linear_model import LogisticRegression
from mord import LogisticIT


def compute_propensity(df, formula, model='logistic', calibrated=True):
    formula = Formula(formula)
    df = formula.get_design_matrix(df)
    
    if model == 'logistic':
        clf = LogisticRegression()
    elif model == 'logisticit':
        clf = LogisticIT()
    
    dm = Formula(formula).get_design_matrix(df)
    clf.fit(dm.rhs, dm.lhs)
    propensity = clf.predict(dm.rhs)

    if model == 'logisticit':
        propensity = propensity[:, [-1]]

    return propensity