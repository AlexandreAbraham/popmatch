import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.utils import check_random_state
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors


def perform_1_to_1_matching(X_source, X_target):
    target_mask = np.ones(X_target.shape[0], dtype=bool)
    target_idx = np.arange(X_target.shape[0])
    nn = []
    nbrs = None
    for sample in X_source:
        i = None
        if nbrs:
            i = nbrs.kneighbors([sample])[1] 
            i = target_idx[target_mask][i]
        
        if i is None or i in nn:
            target_mask[np.asarray(nn)] = False
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(X_target[target_mask])
            i = nbrs.kneighbors([sample])[1] 
            i = target_idx[target_mask][i]

        nn.append(i)
    return np.asarray(nn)


def generate_dataset(n_controls, n_treated, continuous_features, categorical_features,
                     n_informative_propensity, n_informative_outcome,
                     propensity_factor=.5, random_state=None):
    
    n_samples = n_controls + n_treated
    n_features = continuous_features + len(categorical_features)
    n_informative = n_informative_outcome + n_informative_propensity

    # In order to simulate a bias in propensity, we generate a classification problem
    # with errors equal to the propensity factor.

    X_p, y_p = make_classification(
        n_samples=n_samples,
        n_features=n_informative_propensity,
        n_informative=n_informative_propensity,
        n_redundant=0,
        weights=[n_controls / n_samples, n_treated / n_samples],
        flip_y = propensity_factor,
        random_state=random_state,
    )
    
    # Now we generate a regression problem which will be our outcome, but we ignore the
    # labels because we will tweak it afterward.

    X, y, coefs = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative_outcome,
        coef=True,
        random_state=random_state,
    )

    rng = check_random_state(random_state)

    # We replace part of X with data from the classification problem
    X[:, n_informative_outcome:n_informative_outcome + n_informative_propensity] = X_p

    # We add coefs to these features, but with a lesser weight (by default the weight is 100)
    #coefs[n_informative_outcome:n_informative_outcome + n_informative_propensity] = \
    #    30 * rng.uniform(size=(n_informative_propensity))

    # Now we create two coefs table: one for control, one for treated
    coefs_control = coefs.copy()
    coefs_treated = coefs.copy()
    coefs_treated[:n_informative] += 10 * rng.uniform(size=(n_informative   ))

    y = np.zeros(n_samples)
    y[y_p == 0] = np.dot(X[y_p == 0], coefs_control)
    y[y_p == 1] = np.dot(X[y_p == 1], coefs_treated)

    # Turn some features into categorical ones
    idx_cat = rng.choice(n_informative, size=len(categorical_features), replace=False)
    for i, n in zip(idx_cat, categorical_features):
        X[:, i] = np.digitize(X[:, i], np.quantile(X[:, i], (np.arange(n - 1) + 1) / n))

    return {'X': X, 'y': y, 'groups': y_p,
             'idx_cat': idx_cat, 'n_samples': n_samples,
             'n_informative': n_informative}


def generate_pairing_with_errors(dataset, levels, random_state=None):
    X = dataset['X']
    groups = dataset['groups']
    n_informative = dataset['n_informative']
    
    # First, we take each percent of population in order to create the confidence
    # levels.
    percents = [i[0] for i in levels]
    assert(abs(sum(percents) - 1.0) < 0.001)

    # Then, we arbitrarily cluster the samples in order to apply the given error
    # rate in the matching
    rng = check_random_state(random_state)
    rand = rng.uniform((groups == 1).sum())
    true_confidence = groups.copy()
    true_confidence[true_confidence == 1] = np.digitize(rand, np.cumsum(percents)[:-1]) + 1
    fake_confidence = true_confidence.copy()
    corrupted_pairing_candidates = np.arange(X.shape[0])
    corrupted_pairing_candidates[groups == 1] = -1

    for level, (_, error_rate, masked_feature) in zip(range(len(levels), 0, -1), levels):

        # Select a subset of patients corresponding to this level
        to_corrupt = np.where(true_confidence == level)[0]
        to_corrupt = rng.choice(to_corrupt, size=int(error_rate * to_corrupt.shape[0]), replace=False)
        fake_confidence[to_corrupt] = 0  # We turn real paired data into control

        # Randomly drop informative features from data
        X_ = np.delete(X, rng.choice(n_informative, size=masked_feature, replace=False), axis=1)

        # Now we use NN to create fake matches with non paired data
        indices = perform_1_to_1_matching(X_[to_corrupt], X_[corrupted_pairing_candidates != -1])
        fake_confidence[corrupted_pairing_candidates[indices]] = level
        corrupted_pairing_candidates[indices] = -1

    assert((fake_confidence > 0).sum() == groups.sum())

    return fake_confidence        


