from clustering.nec import negentropy_clustering


def NegentropyClustering(centers, clusters, lr, decay_steps, max_epoch, input_shape=None):
    return negentropy_clustering.NEC(centers=centers, clusters=clusters, lr=lr, decay_steps=decay_steps,
                                     max_epoch=max_epoch, input_shape=input_shape)
