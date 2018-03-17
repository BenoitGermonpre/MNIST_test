import tensorflow as tf


def inference(features, mode):
    """Creates the predictions of the model
        Args:
          features (dict): A dictionary of tensors keyed by the feature name.
        Returns:
            A tensor that represents the predictions
    """
    print(type(features['image']))
    print(features['image'].get_shape())

    x = features['image']
    input_layer = tf.reshape(x, [-1, 28, 28, 1], name="Reshape_input")

    K = 6  # first convolutional layer output depth
    L = 12  # second convolutional layer output depth
    M = 24  # third convolutional layer output depth
    N = 200  # fully connected layer

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=K,
        kernel_size=[6, 6],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=L,
        kernel_size=[5, 5],
        strides=(2, 2),
        padding="same",
        activation=tf.nn.relu)

    # Convolutional Layer #2
    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=M,
        kernel_size=[4, 4],
        strides=(2, 2),
        padding="same",
        activation=tf.nn.relu)


    # Dense Layer
    conv3_flat = tf.reshape(conv3, [-1, 7 * 7 * M], name="Reshape_flatten")
    dense1 = tf.layers.dense(inputs=conv3_flat, units=N, activation=tf.nn.relu)
    dropout1 = tf.layers.dropout(inputs=dense1, rate=0.25, training=mode==tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout1, units=10)

    predictions = {
        "logits": logits,
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    print(predictions)
    return predictions


def loss(predictions, labels):
    """Function that calculates the loss and accuracy based on the predictions/logits and labels
        Args:
          predictions: A tensor representing the predictions (output from)
          labels: A tensor representing the labels.
        Returns:
            A tensor representing the loss
    """
    with tf.name_scope("loss"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=predictions['logits'], labels=labels)
        cross_entropy = tf.reduce_mean(cross_entropy)

    return cross_entropy


def build_model_fn():
    """Build model function as input for estimator.
    Returns:
        function: model function
    """

    def _model_fn(features, labels, mode, params):
        """Creates the prediction and its loss.
        Args:
          features (dict): A dictionary of tensors keyed by the feature name.
          labels: A tensor representing the labels.
          mode: The execution mode, defined in tf.estimator.ModeKeys.
        Returns:
          tf.estimator.EstimatorSpec: EstimatorSpec object containing mode,
          predictions, loss, train_op and export_outputs.
        """
        # Get the predictions
        predictions = inference(features, mode)
        loss_op = None
        train_op = None
        metrics = None

        # If we are not in PREDICT mode, we also need the loss
        if mode != tf.estimator.ModeKeys.PREDICT:
            loss_op = loss(predictions, labels)
            accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, 1),
                                           predictions=tf.argmax(predictions["logits"], 1))
            metrics = {
                        "accuracy": accuracy
                        }

        # If we are in TRAIN mode, we need to perform a training step
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss_op,
                global_step=tf.train.get_global_step(),
                learning_rate=params['learning_rate'],
                optimizer='Adagrad',
                summaries=[
                    'learning_rate',
                    'loss',
                    'gradients',
                    'gradient_norm',
                ],
                name='train')

        # Create dict for predictions
        predictions_dict = {"predictions": predictions["probabilities"]}
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                tf.estimator.export.PredictOutput(predictions_dict)}

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions_dict,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops=metrics,
            export_outputs=export_outputs)

    return _model_fn