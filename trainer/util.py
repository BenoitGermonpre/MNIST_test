import tensorflow as tf


def build_input_fn(filenames, batch_size=100, shuffle=False):

    def _parser(record):

        # Dict with the data-names and types we expect
        features = {
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
                }

        # Parse and decode serialized data
        parsed_record = tf.parse_single_example(record, features)
        image = tf.decode_raw(parsed_record['image_raw'], tf.float32)
        label = tf.cast(parsed_record['label'], tf.int32)

        return image, tf.one_hot(label, depth=10)

    def _input_fn():

        # Create TF Dataset object for reading and shuffling data from TFRecords files
        dataset = tf.data.TFRecordDataset(filenames).map(_parser)

        # Shuffle data (optionally)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)

        # Infinite iterations: let training spec determine num_epochs
        #dataset = dataset.repeat(None)

        # Determine batch size and iterate
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        images, labels = iterator.get_next()

        # Put input features in a dict
        features = {
            'image': images
                }

        return features, labels

    return _input_fn


