import tensorflow as tf


def train(model, train_data, settings, log_dir, with_labels=False):
    """
    Function used to train the model, it is able to save training stats and model checkpoint

    :param model: Tensorflow model
    :type model: tf.keras.Model
    :param train_data: Training data
    :type train_data: np.ndarray | tf.data.dataset
    :param settings: Model settings
    :type settings: easydict or dictionary with attribute dot access
    :param log_dir: Path where save model checkpoints and training stats
    :type log_dir: str
    :param with_labels: A value to indicate if labels are present in data or not
    :type with_labels: bool
    :return: None
    """
    # model optimizer and loss
    model.compile(loss=settings.MODEL.TRAIN.LOSS,
                  optimizer=settings.MODEL.TRAIN.OPTIMIZER,
                  metrics=settings.MODEL.TRAIN.METRICS)

    model.summary()

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir, update_freq=10)
    mc_callback = tf.keras.callbacks.ModelCheckpoint(log_dir, verbose=1)
    es_callback = tf.keras.callbacks.EarlyStopping(patience=5)

    train_x = train_data[0]
    if with_labels:
        train_y = train_data[1]
    else:
        train_y = train_data[0]

    # train the model
    model.fit(train_x,
              train_y,
              epochs=settings.MODEL.TRAIN.EPOCHS,
              initial_epoch=settings.MODEL.TRAIN.RESUME_EPOCH,
              batch_size=settings.MODEL.TRAIN.BATCH_SIZE,
              validation_split=settings.MODEL.TRAIN.VALIDATION_SPLIT,
              callbacks=[tb_callback, mc_callback, es_callback],
              shuffle=True)


def evaluate(model, test_data, with_labels=False):
    """

    Function to evaluate trained models

    :param model: Tensorflow model
    :type model: tf.keras.Model
    :param test_data: Data to test on model
    :type test_data: np.ndarray | tf.data.dataset
    :param with_labels: A value to indicate if labels are present in data or not
    :type with_labels: bool
    :return: None
    """

    model.trainable = False
    test_x = test_data[0]

    if with_labels:
        test_y = test_data[1]
    else:
        test_y = test_data[1]

    loss, acc = model.evaluate(test_x, test_y)
    print(f"Evaluation > Loss: {loss} - Acc: {acc}")
