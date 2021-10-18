import numpy as np
from sklearn.preprocessing import StandardScaler


def fisher_criterion(centroids, centroids_mean):
    centroid_1, centroid_2 = centroids
    mc1, mc2 = centroids_mean

    sb = (mc2 - mc1) @ (mc2 - mc1).T  # between-class covariance matri
    sw = np.cov(centroid_1) + np.cov(centroid_2)  # total within-class co- variance matrix.

    w = np.linalg.pinv(sw) @ (mc2 - mc1).T
    jw = (w.T * sb @ w) / (w.T @ sw @ w)

    return w, jw


def negentropy(x, centroids, func, a=10):
    centroid_1, centroid_2 = centroids
    mc1, mc2 = centroids

    if centroid_1.shape[1] == 1 or centroid_2.shape[1] == 1:
        if centroid_1.shape[1] != 1 and centroid_2.shape[1] != 1:
            if centroid_2.shape[1] == 1:
                mc1 = np.mean(centroid_1, axis=1)
            else:
                mc2 = np.mean(centroid_2, axis=1)
    else:
        mc1 = np.mean(centroid_1, axis=1)
        mc2 = np.mean(centroid_2, axis=1)

    w, jw = fisher_criterion(centroids, (mc1, mc2))

    ic1 = np.expand_dims(w @ x, axis=1)

    x = StandardScaler().fit_transform(ic1)
    neg = 0.0
    if func == 1:
        k1 = 36 / (8 * np.sqrt(3) - 9)
        ka2 = 1 / (2 - (6 / np.pi))

        A = np.mean(x * np.exp(-x**2 / 2))**2
        B = (np.mean(np.abs(x)) - np.sqrt(2 / np.pi))**2
        neg = 0.001 * jw + 10 * (k1 * A + ka2 * B)
    elif func == 2:
        neg = (np.mean(x**3))**2 + ((np.mean(np.mean(1/4 * x**4)) -
                                     np.mean(np.mean(1/4 * (np.random.normal(x.shape))**4)))**2)
        neg = np.sqrt(np.sum(neg**2))
    elif func == 3:
        neg = (np.mean((np.mean(- 1 / a * np.exp(- a * x**2 / 2))) -
                       np.mean(np.mean(- 1 / a * np.exp(-a * np.random.normal(x.shape)**2 / 2))))**2)
        neg = np.sqrt(np.sum(neg**2))

    return neg
