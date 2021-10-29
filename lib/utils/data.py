import os
import glob
import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
import joblib


def read_dataset(data_path, columns, split, fillnan=True, filter_labels=None, verbose_name=None):
    """
    :param data_path: Path to dataset directory
    :type data_path: str
    :param columns: Columns that have to be read from csv files
    :type columns: list-like
    :param split: A string to identify if it is a training/validation/test split
    :type split: str
    :param fillnan: Replace NaN values with 0.0 or not
    :type fillnan: bool
    :param filter_labels: Preserves rows with declared labels
    :type filter_labels: list-like
    :param verbose_name: Output name during loading operation
    :type verbose_name: str
    :return: Array with size (number_of_samples, number_of_columns)
    :rtype: Pandas DataFrame
    """

    files = glob.glob(os.path.join(data_path, '**', '*.csv.gz'), recursive=True)
    df = pd.concat([pd.read_csv(file, usecols=columns)
                    for file in tqdm.tqdm(files, desc=f'Dataloader: {verbose_name}')
                    if any(t in file for t in split)], ignore_index=True)
    print(df.head(5))

    if fillnan:
        df.fillna(0.0, inplace=True)

    if filter_labels is not None:
        df = df.loc[df['label'].isin(filter_labels)]

    return np.array(df)


def dataloader(data_path, columns, train_split, test_split=None, fillnan=True, filter_labels=None):
    """

    Read data from disk and concat values all togheter

    :param data_path: Path to data directory
    :type data_path: str
    :param columns: Columns that have to be read from csv files
    :type columns: list-like
    :param train_split: Filename suffix to use as train set
    :type train_split: str
    :param test_split: Filename suffix to use as test set (can be None)
    :type test_split: str
    :param fillnan:
    :type fillnan: bool
    :param filter_labels:
    :type filter_labels: list-like
    :return: Two list [train/test] with dataframe and filename
    :rtype: List[Tuple[Dataframe, str]], List[Tuple[Dataframe, str]]
    """

    if train_split is None and test_split is None:
        raise RuntimeError('One of train_split or test_split must be not none')

    train_data = read_dataset(data_path, columns, train_split, fillnan, filter_labels, 'train data'.upper()) \
        if train_split is not None else None
    test_data = read_dataset(data_path, columns, test_split, fillnan, filter_labels, 'test data'.upper()) \
        if test_split is not None else None

    return train_data, test_data


def read_dataset_list(data_path, columns, split, fillnan=True, filter_labels=None, verbose_name=None):
    """
    Read dataset and return a list of dataframes with associated filename.

    :param data_path: Path to dataset directory
    :type data_path: str
    :param columns: Columns that have to be read from csv files
    :type columns: list-like
    :param split: A string to identify if it is a training/validation/test split
    :type split: str
    :param fillnan: Replace NaN values with 0.0 or not
    :type fillnan: bool
    :param filter_labels: Preserves rows with declared labels
    :type filter_labels: list-like
    :param verbose_name: Output name during loading operation
    :type verbose_name: str
    :return: A list of arrays of shape (number_of_samples, all_columns) and filenames
    :rtype: (list of Pandas DataFrame, str)
    """
    files = glob.glob(os.path.join(data_path, '**', '*.csv.gz'), recursive=True)
    df = [(pd.read_csv(file, usecols=columns), file)
          for file in tqdm.tqdm(files, desc=f'Dataloader/file: {verbose_name}')
          if any(t in file for t in split)]

    if filter_labels:
        df = [(dfs.loc[dfs['label'].isin(filter_labels)], dfn) for dfs, dfn in df]

    if fillnan:
        df = [(dfs.fillna(0.0, inplace=False), dfn) for dfs, dfn in df]

    return df


def dataloader_single(data_path, columns, train_split, test_split=None, fillnan=True, filter_labels=None):
    """

    Read data from disk and return a list for each files with their name, it does not concat values all togheter

    :param data_path: Path to data directory
    :type data_path: str
    :param columns: Columns that have to be read from csv files
    :type columns: list-like
    :param train_split: Filename suffix to use as train set
    :type train_split: str
    :param test_split: Filename suffix to use as test set (can be None)
    :type test_split: str
    :param fillnan:
    :type fillnan: bool
    :param filter_labels:
    :type filter_labels: list-like
    :return: Two list [train/test] with dataframe and filename
    :rtype: List[Tuple[Dataframe, str]], List[Tuple[Dataframe, str]]
    """

    train_data = read_dataset_list(data_path, columns, train_split, fillnan, filter_labels, 'train data'.upper())
    test_data = read_dataset_list(data_path, columns, test_split, fillnan, filter_labels, 'test data'.upper()) \
        if test_split is not None else None

    return train_data, test_data


def label_mapping(y_data, known_labels):
    """
    Maps all class index to used one. It is possible to subsample number of classes.

    :param y_data: Array with labels to map
    :type y_data: nd.array
    :param known_labels: Labels index to map
    :type known_labels: list-like
    :return: label_map and y_data with cls_idx remapped
    :rtype: tuple(list, nd.array)
    """
    label_map = {}
    for i, l in enumerate(known_labels):
        label_map[i] = l

    for i, l in enumerate(known_labels):
        print(f"\t {l} => {i}")
        y_data[y_data == l] = i

    return label_map, y_data


def split_labels(data, label_idx=-1):
    """
    Split labels from numerical data

    :param data: array of inputs data
    :type data: nd.array
    :param label_idx: index where label is located in the array. It can be only at start of at the end of the array
    :type label_idx: int
    :return: data without labels, labels
    :rtype: nd.array, nd.array
    """
    # return (Data, Labels)
    if label_idx == -1 or label_idx == data.shape[-1]:
        return data[..., :-1], data[..., -1]
    elif label_idx == 0:
        return data[..., 1:], data[..., 0]
    else:
        raise RuntimeError('Labels must be on axis 0 or 1')


def preprocessing(data, scaler, settings, log_dir, fit_scaler=False, with_labels=False):
    """

    :param data: Data array
    :type data: np.ndarray
    :param settings: Settings dictionary
    :type settings: easydict or dict with dot attribute access
    :param log_dir: Path where save scaler
    :type log_dir: str
    :param fit_scaler: Used to load or create a new scaler
    :type fit_scaler: bool
    :param with_labels: used to know if labels are used in the dataset or not
    :type with_labels: bool
    :return: Scaler and two lists with train/test information
    :rtype: sklearn Scaler, List[np.ndarray, np.ndarray], List[np.ndarray, np.ndarray]
    """
    if with_labels:
        train_data = data[0][0]
        train_label = data[0][1]

        test_data = data[1][0]
        if test_data is not None:
            test_label = data[1][0]
        else:
            test_label = None

    else:
        train_data = data[0]
        train_label = None

        test_data = data[1]
        test_label = None

    if fit_scaler:
        scaler = scaler.fit(train_data)
        joblib.dump(scaler, os.path.join(log_dir, 'scaler.joblib'))

    train_data = scaler.transform(train_data)
    if test_data is None:
        # train test split
        if with_labels:
            train_data, test_data, train_label, test_label = train_test_split(train_data, train_label,
                                                                              test_size=settings.DATASET
                                                                              .PREPROCESSING.VALIDATION.TEST_SIZE,
                                                                              random_state=settings.DATASET
                                                                              .PREPROCESSING.RANDOM_SEED)
        else:
            train_data, test_data = train_test_split(train_data,
                                                     test_size=settings.DATASET.PREPROCESSING.VALIDATION.TEST_SIZE,
                                                     random_state=settings.DATASET.PREPROCESSING.RANDOM_SEED)
    else:
        test_data = scaler.transform(test_data)

    return [train_data, train_label], [test_data, test_label]


def get_scaler(settings, log_dir, load_scaler=False):
    if settings.DATASET.PREPROCESSING.SCALER.TYPE == 'standard':
        scaler = StandardScaler()  # Range is not defined, best to use a StandardScaler
    elif settings.DATASET.PREPROCESSING.SCALER.TYPE == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = FunctionTransformer()

    if load_scaler:
        scaler = joblib.load(settings.DATASET.PREPROCESSING.SCALER.LOAD)

    return scaler
