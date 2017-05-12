from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
# import pandas as pd


class KnnAD:
    """
    The underlying assumption is that K >> #anomalies, so that anomalies will have to connect with nominal points, thus
    creating larger average distances than the nominal points.
    """
    def __init__(self, K=None):
        self.K = K + 1 if K is not None else K      # use K+1 since first neighbour is itself

    def detect(self, data, K=None, pca_dim=None):
        X = data
        if pca_dim:
            X = PCA(n_components=pca_dim).fit_transform(data)
        nbrs = NearestNeighbors(n_neighbors=K+1 or self.K, algorithm="auto").fit(X)
        distances, _ = nbrs.kneighbors(return_distance=True)
        # result = pd.DataFrame(data=distances[:, 1:], index=idx[:, 0])
        distances = distances[:, 1:]                # remove the closest neighbor which is self
        result = distances.mean(axis=1)
        result /= result.sum()
        return result


class LogLikelihood:
    """
    H_0 : The samples are coming from data distribution.
    H_1 : The samples are coming from anomaly distribution
    """
    pass


# TODO: isolated random forest: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html


