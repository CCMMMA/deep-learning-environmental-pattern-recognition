from clustering.nec import negentropy_clustering


def NegentropyClustering(n_centers, lr, decay_steps, max_epoch, input_shape=None):
    return negentropy_clustering.NEC(n_centers=n_centers, lr=lr, decay_steps=decay_steps,
                                     max_epoch=max_epoch, input_shape=input_shape)
