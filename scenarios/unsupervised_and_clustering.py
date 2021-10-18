import __init_paths
import tensorflow as tf
from tensorflow.keras import Model
import os
import argparse
from datetime import datetime
from easydict import EasyDict as edict
import numpy as np

import yaml

from models import unsupervised, clustering
import utils
import h5py


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=str)

    return parser.parse_args()


def main(args):
    print("==== Phase 0. Loading settings")
    settings = args.settings
    with open(settings, 'r') as file:
        settings = yaml.load(file, Loader=yaml.FullLoader)

    settings = edict(settings)

    print("==== Phase 0. Dataloader")
    # Data loading and preprocessing
    data = utils.data.dataloader(settings.DATASET.PATH, settings.DATASET.COLUMNS,
                                 train_split=settings.DATASET.TRAINING,
                                 test_split=settings.DATASET.TESTING)

    weights = False
    if settings.GLOBAL.RESUME_PATH:
        log_dir = settings.GLOBAL.RESUME_PATH
        weights = log_dir
    else:
        log_dir = os.path.join(settings.GLOBAL.SAVE_PATH, settings.MODEL.NAME, datetime.now().strftime('%Y%m%d-%H%M%S'))
        print(f'Reference path: {log_dir}')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    print("==== Phase 0. Preprocessing")
    with_labels = settings.DATASET.IS_LABEL  # Check if dataset contains labels
    # Preprocess data with selected scaler
    scaler, train_data, test_data = utils.data.preprocessing(data, settings, log_dir,
                                                             load_scaler=weights, with_labels=with_labels)
    # Get number of output features
    n_features = train_data[0].shape[-1]

    print(f"==== Phase 1. Building model {settings.MODEL.NAME}")
    # Section 1.
    autoencoder = unsupervised.__dict__[settings.MODEL.NAME](settings.MODEL.ENCODING_DIMS, n_features)

    # Check if model has to be loaded
    if weights:
        autoencoder = tf.keras.models.load_model(weights)
        print(f'Restored from: {weights}')
    # Check if pipeline is in training or evaluation mode
    if settings.MODEL.MODE == 'train':
        print("==== Phase 1. Model Training")
        utils.model.train(autoencoder, train_data, settings, log_dir, with_labels=with_labels)
    elif settings.MODEL.MODE == 'eval':
        print("==== Phase 1.  Model Evaluation")
        utils.model.evaluate(autoencoder, scaler, test_data, with_labels=with_labels)

    # Section 2.
    print("==== Phase 2. Feature Extraction")
    # Avoid autoencoder to be trainable
    autoencoder.trainable = False

    # Build a new Model class with the encoder part only
    feature_extractor = Model(inputs=autoencoder.input,
                              outputs=autoencoder.get_layer('encoder').output)

    train_features = feature_extractor.predict(train_data[0])

    print(f"==== Phase 2. Clustering: {settings.CLUSTERING.NAME}")
    nec_clustering = clustering.__dict__[settings.CLUSTERING.NAME](n_centers=settings.CLUSTERING.N_CENTERS,
                                                                   lr=settings.CLUSTERING.LR,
                                                                   decay_steps=settings.CLUSTERING.DECAY_STEPS,
                                                                   max_epoch=settings.CLUSTERING.MAX_EPOCH)
    print("==== Phase 2. Clustering fitting")
    nec_clustering.fit(train_features)
    train_ng, train_clusters = nec_clustering.predict(train_features)

    print("==== Phase 2. Clustering test predict")
    test_features = feature_extractor.predict(test_data)
    test_ng, test_clusters = nec_clustering.predict(test_features)

    print("=== Phase 3. Save clustering")
    with h5py.File(os.path.join(settings.GLOBAL.SAVE_PATH, 'train_clusters.h5'), 'w') as df:
        df.create_dataset('train', shape=train_clusters.shape, data=train_clusters)
        df.create_dataset('test', shape=test_clusters.shape, data=test_clusters)


if __name__ == '__main__':
    args = args_parse()
    main(args)
