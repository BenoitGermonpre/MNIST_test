import tensorflow as tf
from trainer.util import build_input_fn
from trainer.model import build_model_fn
from trainer.config import MODEL_DIR, TFRECORD_DIR, BATCH_SIZE
#from archive.config_local import MODEL_DIR, TFRECORD_DIR, BATCH_SIZE
import os, glob


if __name__ == '__main__':

    # Create estimator instance
    estimator = tf.estimator.Estimator(
        model_fn=build_model_fn(),
        model_dir=MODEL_DIR,
        params={'learning_rate': 0.0025})

    # Build input functions for training and testing
    train_input_fn = build_input_fn(glob.glob(os.path.join(TFRECORD_DIR, 'train-*.tfrecords')), batch_size=BATCH_SIZE)
    test_input_fn = build_input_fn(os.path.join(TFRECORD_DIR, 'test.tfrecords'), batch_size=BATCH_SIZE)

    # Prepare spec for train and test data
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=1000)
    eval_spec = tf.estimator.EvalSpec(input_fn=test_input_fn, steps=500)

    # Train and evaluate estimator
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # Prepare for Serving
    feature_placeholders = {
            'image': tf.placeholder(tf.float32, [None, 784], name='img_placeholder'),
        # label is not required since serving is only used for inference
    }

    serving_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
        feature_placeholders)

    # Export model
    estimator.export_savedmodel(
        MODEL_DIR, serving_input_fn)