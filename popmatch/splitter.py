import numpy as np
from sklearn.utils import check_random_state
from tqdm import tqdm


class GeneralPurposeClustering:
    
    def __init__(self, n_clusters, loss, n_iter=10000, verbose=0, random_state=None):
        if isinstance(self.n_clusters, int):
            self.n_clusters = n_clusters
            self.p_clusters = None
        elif isinstance(self.n_clusters, list):
            self.n_clusters = len(n_clusters)
            self.p_clusters = n_clusters
        self.n_iter = n_iter
        self.loss = loss
        self.verbose = verbose
        self.rng = check_random_state(random_state)
        
    def fit(self, X):
        cluster_id = self.rng.random.choice(self.n_clusters, replace=True, size=X.shape[0], p=self.p_clusters)
        current_loss = self.loss(X, cluster_id)

        range_iter = tqdm(range(self.n_iter)) if self.verbose >= 1 else range(self.n_iter)
        if self.verbose >= 1:
            losses = []
        for i in range_iter:
            ci, cj = self.rng.random.choice(self.n_clusters, size=2)
            i = self.rng.random.choice(np.where(cluster_id == ci)[0])
            j = self.rng.random.choice(np.where(cluster_id == cj)[0])

            # Swap sample clusters and recompute loss
            cluster_id[i], cluster_id[j] = cluster_id[j], cluster_id[i]
            new_loss = self.loss(X, cluster_id)

            if new_loss > current_loss:
                cluster_id[i], cluster_id[j] = cluster_id[j], cluster_id[i]
            else:
                current_loss = new_loss
                if self.verbose >= 1:
                    losses.append(i, current_loss)

        if self.verbose >= 1:
            self.loss_ = losses
        self.cluster_id_ = cluster_id
