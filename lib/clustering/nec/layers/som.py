import numpy as np
import tqdm


class SOM(object):
    def __init__(self, centers, lr, decay_steps, input_shape=None, verbose=True):
        """
        SOM with Winner take all approach
        :param centers: number of centroids to find
        :param lr: initial learning rate
        :param decay_steps: learning rate decay steps
        :param verbose: Verbose output
        :param input_shape: tuple with (# inputs, # patterns)
        """
        self.centers = centers
        self.decay_steps = decay_steps
        self.lr = lr
        self.verbose = not verbose if (verbose or verbose is not None) else False
        self.kernel = None

        if input_shape is not None:
            self.inputs = input_shape[0]
            self.patterns = input_shape[1]
            self.built = True
        else:
            self.inputs = None
            self.patterns = None
            self.built = False

    def _build(self, input_shape):
        self.inputs = input_shape[0]
        self.patterns = input_shape[1]
        self.built = True

    def fit(self, x, epochs):
        """
        Fit the SOM model
        :param x: input parameter
        :param epochs: number of epoch to train
        """
        if not self.built:
            self._build(x.shape)

        ep_iterator = tqdm.trange(epochs, disable=self.verbose)
        if self.inputs < self.centers:  # Min difference delta?
            replica_num = np.ceil(self.centers / self.inputs) * np.random.randint(3, 11)
            x_replica = np.tile(x, (int(replica_num), 1))
            perm_centers = np.random.permutation(x_replica.shape[0])
        else:
            x_replica = x
            perm_centers = np.random.permutation(self.inputs)

        self.kernel = x_replica[perm_centers[:self.centers]]
        print("SOM Kernel", self.kernel.shape)
        print("SOM inputs", x.shape)
        print("SOM replica_inputs", x_replica.shape)

        lr = self.lr * np.exp(-np.arange(0, epochs) / self.decay_steps)
        mean_loss = -1
        for k in ep_iterator:
            ep_iterator.set_description(f'Epoch {k+1}/{epochs} - Loss: {mean_loss}')
            mean_loss = 0
            batch_it = tqdm.trange(self.inputs, disable=self.verbose)
            for i in batch_it:
                diff_x_k = x[i] - self.kernel
                loss = np.linalg.norm(diff_x_k, axis=1)
                mx, pos = np.amin(loss), np.argmin(loss)
                dW = lr[k] * diff_x_k[pos]
                self.kernel[pos] = self.kernel[pos] + dW
                mean_loss += mx
            mean_loss = mean_loss / self.inputs

    def predict(self, x, batch_size=0, verbose=None):
        """
        :param x: input to evaluate
        :return: Clusters with evaluation of x
        """
        in_verbose = not verbose if (verbose or verbose is not None) else False
        if batch_size > 0:
            batches = np.array_split(x, batch_size)
            result = []
            el_batches = tqdm.tqdm(batches, total=batches.shape[0], disable=in_verbose)
            for batch in el_batches:
                result.append(self(batch, verbose=verbose))
            return result

        return self(x, verbose=verbose)

    def __call__(self, x, verbose=None):
        """
        :param x: input to evaluate
        :return: Clusters with evaluation of x
        """

        if self.kernel is None:
            raise RuntimeError('Weights are not initialized, please use .fit() before prediction')

        verbose = not verbose if (verbose or verbose is not None) else False

        clusters = [[] for _ in range(self.centers)]
        el_inputs = tqdm.trange(x.shape[0], desc='Predict', disable=verbose)
        inputs_order = []
        for i in el_inputs:
            loss = np.linalg.norm(x[i] - self.kernel, axis=1)
            pos = int(np.argmin(loss))
            clusters[pos].append(x[i])
            inputs_order.append(pos)

        clusters = [np.array(cluster) for cluster in clusters]

        return clusters, inputs_order


def som(x, epochs, n_centers, lr, decay_rate, verbose=True):
    ep_iterator = tqdm.trange(epochs, disable=verbose)

    n_inputs, n_patterns = x.shape

    R = np.random.permutation(n_patterns)

    W = x[:, R[:n_centers]]

    for k in ep_iterator:
        for i in tqdm.trange(n_patterns, disable=verbose):
            lr = lr * np.exp(-k / decay_rate)
            x_c = np.broadcast_to(x[:, i], (n_centers, 2)).T
            loss = np.linalg.norm(x_c - W, axis=0)
            mx, pos = np.amin(loss), np.argmin(loss)
            dW = lr * (x[:, i] - W[:, pos])
            W[:, pos] = W[:, pos] + dW

    return W


def som_cluters(x, W):

    n_clusters = W.shape[1]

    n_inputs, n_patterns = x.shape

    clusters = [np.zeros((2, 0))] * n_clusters

    for i in range(n_patterns):
        x_c = np.broadcast_to(x[:, i], (n_clusters, 2)).T  # (X(i,:).' * ones(1, n_clusters)).'
        loss = np.linalg.norm(x_c - W, axis=0)
        pos = int(np.argmin(loss))  # CHECK AXIS
        clusters[pos] = np.append(clusters[pos], np.reshape(x[:, i], (-1, 1)))

    clusters = [np.reshape(cluster, (2, -1)) for cluster in clusters]

    return clusters


if __name__ == '__main__':
    x = np.random.rand(15, 5)
    s = SOM(50, 0.1, 10, verbose=True)
    print("Fitting")
    s.fit(x, 10)
    x_test = np.random.rand(500, 5)
    print("Testing")
    s.predict(x_test, verbose=True)
