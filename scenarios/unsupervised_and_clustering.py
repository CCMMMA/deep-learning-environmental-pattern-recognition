import __init_paths
import tensorflow as tf
from tensorflow.keras import Model
import os
import argparse
from datetime import datetime
from easydict import EasyDict as edict
import numpy as np
import joblib
import yaml

from models import unsupervised, clustering
import utils
import h5py


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=str)

    return parser.parse_args()


def clusterize(backbone, clustering_model, scaler, settings, log_dir, data=None):
    if not settings.CLUSTERING.LOAD:
        print("==== Phase 2. Feature extraction")
        train_features = backbone.predict(data[0])

        print("==== Phase 2. Clustering fitting")
        clustering_model.fit(train_features)

        if settings.CLUSTERING.SAVE:
            path = settings.CLUSTERING.SAVE
            if not os.path.isabs(settings.CLUSTERING.SAVE):
                path = os.path.join(log_dir, settings.CLUSTERING.SAVE)
            joblib.dump(clustering_model, path)
    else:
        print("==== Phase 2. Clustering load weights")
        print(f"Restore {settings.CLUSTERING.NAME} from: {settings.CLUSTERING.LOAD}")
        clustering_model = joblib.load(settings.CLUSTERING.LOAD)

    print("=== Phase 3. Reload dataset with geographical information and evaluate")
    train_data, test_data = utils.data.dataloader_single(settings.DATASET.PATH,
                                                         settings.DATASET.IDX_COLS +
                                                         settings.DATASET.COORDS_COLS +
                                                         settings.DATASET.COLUMNS,
                                                         train_split=settings.DATASET.TRAINING)

    utils.weather.evaluate_clusters_and_store(train_data, 'all', backbone, clustering_model, scaler,
                                              log_dir)


def build_clustering(backbone, settings):

    # Section 2.
    print("==== Phase 2. Build Feature Extractor")
    # Avoid autoencoder to be trainable
    backbone.trainable = False

    # Build a new Model class with the encoder part only
    backbone = Model(inputs=backbone.input,
                     outputs=backbone.get_layer('encoder').output)

    print(f"==== Phase 2. Clustering: {settings.CLUSTERING.NAME}")
    clustering_model = clustering.__dict__[settings.CLUSTERING.NAME](centers=settings.CLUSTERING.CENTERS,
                                                                     clusters=settings.CLUSTERING.CLUSTERS,
                                                                     lr=settings.CLUSTERING.LR,
                                                                     decay_steps=settings.CLUSTERING.DECAY_STEPS,
                                                                     max_epoch=settings.CLUSTERING.MAX_EPOCH)

    return backbone, clustering_model


def main(args):
    print("==== Phase 0. Loading settings")
    settings = args.settings
    with open(settings, 'r') as file:
        settings = yaml.load(file, Loader=yaml.FullLoader)

    settings = edict(settings)

    model_load = settings.MODEL.LOAD
    if settings.GLOBAL.RESUME_PATH:
        log_dir = settings.GLOBAL.RESUME_PATH
    else:
        log_dir = os.path.join(settings.GLOBAL.SAVE_PATH, settings.MODEL.NAME, datetime.now().strftime('%Y%m%d-%H%M%S'))
        print(f'Reference path: {log_dir}')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    with_labels = settings.DATASET.IS_LABEL  # Check if dataset contains labels
    scaler = utils.data.get_scaler(settings, log_dir, load_scaler=settings.DATASET.PREPROCESSING.SCALER.LOAD)
    data = [None]

    if settings.GLOBAL.MODE in ['train', 'test']:
        print("==== Phase 0. Dataloader")
        # Data loading and preprocessing
        data = utils.data.dataloader(settings.DATASET.PATH, settings.DATASET.COLUMNS,
                                     train_split=settings.DATASET.TRAINING,
                                     test_split=settings.DATASET.TESTING)

        print("==== Phase 0. Preprocessing")
        # Preprocess data with selected scaler
        data = utils.data.preprocessing(data, scaler, settings, log_dir,
                                        fit_scaler=not settings.DATASET.PREPROCESSING.SCALER.LOAD,
                                        with_labels=with_labels)
        train_data, test_data = data
        # Get number of output features
        n_features = train_data[0].shape[-1]

        print(f"==== Phase 1. Building model {settings.MODEL.NAME}")
        # Section 1.
        autoencoder = unsupervised.__dict__[settings.MODEL.NAME](settings.MODEL.ENCODING_DIMS, n_features)

    # Check if model has to be loaded
    if model_load:
        autoencoder = tf.keras.models.load_model(model_load)
        print(f'Restored from: {model_load}')
    # Check if pipeline is in training or evaluation mode
    if settings.MODEL.MODE == 'train':
        print("==== Phase 1. Model Training")
        utils.model.train(autoencoder, train_data, settings, log_dir, with_labels=with_labels)
    elif settings.MODEL.MODE == 'test':
        print("==== Phase 1.  Model Evaluation")
        utils.model.evaluate(autoencoder, scaler, test_data, with_labels=with_labels)

    backbone, clustering_model = build_clustering(autoencoder, settings)
    clusterize(backbone, clustering_model, scaler=scaler, log_dir=log_dir, settings=settings, data=data[0])


if __name__ == '__main__':
    args = args_parse()
    main(args)
