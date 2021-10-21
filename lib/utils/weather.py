import os
import numpy as np
import pandas as pd


def evaluate_and_store(records, set_type, model, x_scaler, label_map, log_dir, filter_labels):
    """
    Evaluate model on weather pattern on a list of data. It automatically save a new file with
    two new columns one with classification numeric result and another with classification label name

    :param records: A list of dataframe
    :type records: List[Dataframe]
    :param set_type: Name to associate with output_file (train|test)
    :type set_type: str
    :param model: Tensorflow model
    :type model: tf.keras.Model
    :param x_scaler: Scaler used in training process
    :type x_scaler: sklearn Scaler
    :param label_map: Label to associate with output
    :type label_map: dict
    :param log_dir: Path to directory where save output files
    :type log_dir: str
    :param filter_labels: Columns to use in estimation process
    :type filter_labels: list-like
    :return: None
    """
    print(f"Evaluate on {set_type} set")
    for dfs, dfn in records:
        y_test = dfs.pop('label')
        y_test = np.expand_dims(y_test, -1)

        coords_idx = [dfs.pop('j'), dfs.pop('i'),
                      dfs.pop('lat'), dfs.pop('lon')]

        df_coord = pd.concat(coords_idx, axis=1)

        for i, l in enumerate(filter_labels):
            y_test[y_test == l] = i

        x_test = x_scaler.transform(dfs)

        test_loss, test_acc = model.evaluate(x_test, y_test)
        print(f"Test {dfn} > loss: {test_loss} - acc: {test_acc}")

        y_logits = model.predict(x_test)
        y_pred = np.argmax(y_logits, axis=-1)

        y_pred_map = []
        true_labels = []
        for pred, yt in zip(y_pred, y_test):
            y_pred_map.append(label_map[int(pred)])
            true_labels.append(label_map[int(yt)])

        df_coord['predictions'] = y_pred_map
        df_coord['true_labels'] = true_labels
        filename = dfn.split('/')[-1]
        df_coord.to_csv(os.path.join(log_dir, f"{set_type}_out_{filename}"), index=False)


def evaluate_clusters_and_store(records, set_type, feature_extractor, model, x_scaler, log_dir):
    """
    Evaluate model on weather pattern on a list of data. It automatically save a new file with
    two new columns one with classification numeric result and another with classification label name

    :param records: A list of dataframe
    :type records: List[Dataframe]
    :param set_type: Name to associate with output_file (train|test)
    :type set_type: str
    :param model: Tensorflow model
    :type model: tf.keras.Model
    :param x_scaler: Scaler used in training process
    :type x_scaler: sklearn Scaler
    :param label_map: Label to associate with output
    :type label_map: dict
    :param log_dir: Path to directory where save output files
    :type log_dir: str
    :param filter_labels: Columns to use in estimation process
    :type filter_labels: list-like
    :return: None
    """
    print(f"Evaluate on {set_type} set")
    for dfs, dfn in records:

        coords_idx = [dfs.pop('j'), dfs.pop('i'),
                      dfs.pop('lat'), dfs.pop('lon')]

        df_coord = pd.concat(coords_idx, axis=1)

        x_test = x_scaler.transform(dfs)

        x_test = feature_extractor.predict(x_test)
        y_clusters = model.predict(x_test)

        df_coord['clusters'] = y_clusters
        filename = dfn.split('/')[-1]
        df_coord.to_csv(os.path.join(log_dir, f"{set_type}_out_{filename}"), index=False)
