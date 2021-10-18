import __init_paths
import tensorflow as tf
import os
import argparse
from datetime import datetime
from easydict import EasyDict as edict
import yaml
import numpy as np
from sklearn.metrics import mean_squared_error

from models import classification
import utils


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
                                 test_split=settings.DATASET.TESTING,
                                 filter_labels=settings.DATASET.KNOWN_LABELS)

    # Splits labels from loaded data
    data = (utils.data.split_labels(data[0]), utils.data.split_labels(data[1]))

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
    with_labels = settings.DATASET.IS_LABEL
    scaler, train_data, test_data = utils.data.preprocessing(data, settings, log_dir,
                                                             load_scaler=weights, with_labels=with_labels)

    # Get number of classes
    n_labels = len(np.unique(train_data[1]))
    print(f"==== Phase 1. Building model {settings.MODEL.NAME}")
    # Section 1.
    model = classification.__dict__[settings.MODEL.NAME](train_data[0].shape[1:], n_labels)

    print("==== Phase 1. Label Mapping")
    label_map, train_data[1] = utils.data.label_mapping(train_data[1], settings.DATASET.KNOWN_LABELS)

    # Check if model has to be loaded
    if weights:
        model = tf.keras.models.load_model(weights)
        print(f'Restored from: {weights}')
    # Check if pipeline is in training or evaluation mode
    if settings.MODEL.MODE == 'train':
        print("==== Phase 1. Model Training")
        utils.model.train(model, train_data, settings, log_dir, with_labels=with_labels)
    elif settings.MODEL.MODE == 'eval':
        print("==== Phase 1.  Model Evaluation")
        test_data_relabel = test_data[1]
        for i, l in enumerate(settings.DATASET.KNOWN_LABELS):
            test_data_relabel[1][test_data[1] == l] = i

        utils.model.evaluate(model, scaler, test_data, with_labels=with_labels)

    # Section 2.
    print("=== Phase 2. Predict and store")
    train_data, test_data = utils.data.dataloader_single(settings.DATASET.PATH,
                                                         settings.DATASET.IDX_COLS +
                                                         settings.DATASET.COORDS_COLS +
                                                         settings.DATASET.COLUMNS,
                                                         train_split=settings.DATASET.TRAINING,
                                                         test_split=settings.DATASET.TESTING)

    utils.weather.evaluate_and_store(train_data, 'training', model, scaler,
                                     label_map, log_dir, settings.DATASET.KNOWN_LABELS)
    utils.weather.evaluate_and_store(test_data, 'test', model, scaler,
                                     label_map, log_dir, settings.DATASET.KNOWN_LABELS)


if __name__ == '__main__':
    args = args_parse()
    main(args)
