import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import numpy as np


def plot_clusters(x, c, w):
    colors = ['g', 'r', 'c', 'm', 'y', 'k']
    signs = ['.', 'o', 'x', '+', 'x', 's', 'd', 'v', '^', '<', '>', 'p', 'h']

    sc = []
    for color in colors:
        for sign in signs:
            sc.append(color + sign)

    n_clusters = w.shape[1]
    rs = np.random.permutation(len(sc))
    rsx = sc


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # # create the counts of samples under each node
    # counts = np.zeros(model.children_.shape[0])
    # n_samples = len(model.labels_)
    # for i, merge in enumerate(model.children_):
    #     current_count = 0
    #     for child_idx in merge:
    #         if child_idx < n_samples:
    #             current_count += 1  # leaf node
    #         else:
    #             current_count += counts[child_idx - n_samples]
    #     counts[i] = current_count
    #
    # linkage_matrix = np.column_stack([model.children_, model.distances_,
    #                                   counts]).astype(float)

    # Plot the corresponding dendrogram
    plt.title('Hierarchical Clustering Dendrogram')
    dendrogram(model, **kwargs)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()
