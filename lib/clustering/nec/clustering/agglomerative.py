import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from clustering.nec.losses import negentropy
from clustering.nec.plotting import plot_dendrogram


def agglomerative(centroids, t=0.5, criterion='distance', method='single', metric='euclidean', plot=False):
    """
    Agglomerative clustering of centroids based on negentropy

    :param centroids: Centroids to clusterize
    :param t: threshold value for criterion, based on
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html
    :param criterion: criterion to assign a value to a cluster, base on
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html
    :param method: linkage method default 'single', based on
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage
    :param metric: comparison method default 'euclidean', based on
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage
    :param plot: Plot dendrogram
    :return: negentropy and clustering
    """

    n_centroids = len(centroids)

    ng = []

    for i in range(n_centroids):
        for j in range(i+1, n_centroids):
            centroid = np.concatenate([centroids[i], centroids[j]], axis=0)
            ng.append(negentropy(centroid, [centroids[i], centroids[j]], func=1, a=1))

    ng = np.asarray(ng)
    Z = linkage(ng, method=method, metric=metric)

    if plot:
        plot_dendrogram(Z)

    clustering = fcluster(Z, t=t, criterion=criterion)
    return ng, clustering
