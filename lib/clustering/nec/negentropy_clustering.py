import numpy as np
from scipy.io import loadmat
from clustering.nec.layers import som, som_cluters, SOM
from clustering.nec.clustering import agglomerative
from absl import app
from absl import flags
import matplotlib.pyplot as plt

flags.DEFINE_string("data_path", "/Users/emanueldinardo/Downloads/nec_dendogramma/DD.mat", "Path to mat test file")
flags.DEFINE_string("mat_key", "DD", "Path to mat test file")

FLAGS = flags.FLAGS


class NEC(object):
    """
    Negentropy Clustering
    """
    def __init__(self, n_centers, lr=0.1, decay_steps=1000, max_epoch=100, input_shape=None):
        """

        :param n_centers: Number of centers/clusters
        :type n_centers: int
        :param lr: Learning rate
        :type lr: float
        :param decay_steps: Decay steps to use with learning rate
        :type decay_steps: int
        :param max_epoch: Number of training steps
        :type max_epoch: int
        :param input_shape: Shape of the input data, can be None
        :type input_shape: Tuple(int, int)
        """
        self.n_centers = n_centers
        self.lr = lr
        self.decay_steps = decay_steps
        self.max_epoch = max_epoch

        self.maps = SOM(centers=n_centers, lr=lr, decay_steps=decay_steps, verbose=True, input_shape=input_shape)
        if input_shape is not None:
            self.built = True
        else:
            self.built = False
        self.centers = 0

    def fit(self, inputs):
        self.maps.fit(inputs, epochs=self.max_epoch)
        self.centers = self.maps(inputs)
        self.built = True

    def __call__(self, inputs):
        if not self.built:
            raise RuntimeError('Please use .fit() before call')
        _, clusters = agglomerative(self.centers, plot=True)
        return clusters

    def predict(self, inputs):
        return self(inputs)


def main(_):
    D = loadmat(FLAGS.data_path)[FLAGS.mat_key].T

    plt.plot(D[0, :], D[1, :], 'bo')
    plt.show()

    n_centers = 50

    x = D

    lr = 0.1
    decay_steps = 1000
    maxepoch = 1000

    self_maps = SOM(input_dim=x.shape, centers=n_centers, lr=lr, decay_steps=decay_steps, verbose=True)
    self_maps.fit(x, epochs=maxepoch)
    centers = self_maps(x)

    # OR using functions
    # weights = som(x, maxepoch, n_centers, lr, decay_rate, verbose=True)
    # centers = som_cluters(x, weights)

    # Cplot(X,Center,W);

    # Hierarchical Cluatering based on Negentropy

    ng, clusters = agglomerative(centers, plot=True)
    print(ng)
    print(clusters)


if __name__ == '__main__':
    app.run(main)
