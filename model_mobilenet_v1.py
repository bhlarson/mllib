from mobilenet_v1 import make_mobilenet_v1
from augmentations.py import augment

def loss(logits, labels, num_classes):
    #one_hots = tf.one_hot(labels, num_classes)
    #weights = tf.reduce_sum(one_hots, axis=3)

    max_x_ent_loss = tf.reduce_mean((tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)))
    return max_x_ent_loss


def mobilenet_v1_model_fn(features, labels, mode, params = {height = 512, width=512,data_format="NHWC"}):

    images = features["image"]
    images = tf.image.convert_image_dtype(images, tf.float32)
    images = tf.cast(
        tf.map_fn(preprocessing.mean_image_addition, features),
        tf.float32)
    
    if data_format == 'NHWC':
        # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
        # This provides a large performance boost on GPU. See
        # https://www.tensorflow.org/performance/performance_guide#data_formats
        images = tf.transpose(images, [0, 2, 1, 3], name="nhwc_input")

    # Augment dataset when training
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        images = augment(images, params)

    images = tf.image.per_image_standardization(images)

    images = tf.image.resize_with_crop_or_pad(images,params.height,params.width)

    # make_mobilenet_v1
    logits = make_mobilenet_v1(images, mode)  

    predictions = {
        # Returns the highest prediction from the output of logits.
        tf.contrib.learn.PredictionKey.CLASSES : tf.argmax(logits, axis=1, name="predicted_classes"),
        # Softmax squashes arbitrary real values into a range of 0 to 1 and returns the
        # probabilities of each of the classes. Name is used for logging.
        tf.contrib.learn.PredictionKey.PROBABILITIES : tf.nn.softmax(logits, name="predicted_probability")
    }

    # Compute predictions.
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    labels = tf.squeeze(labels, axis=3)  # reduce the channel dimension.

    logits_by_num_classes = tf.reshape(logits, [-1, params['num_classes']])
    labels_flat = tf.reshape(labels, [-1, ])
    valid_indices = tf.to_int32(labels_flat <= params['num_classes'] - 1)
    valid_logits = tf.dynamic_partition(logits_by_num_classes, valid_indices, num_partitions=2)[1]
    valid_labels = tf.dynamic_partition(labels_flat, valid_indices, num_partitions=2)[1]

    pred_loss = loss(valid_logits, valid_labels, params['num_classes'])

    #accuracy = tf.metrics.accuracy(labels=labels,predictions=predictions,name='acc_op')
    #accuracy = tf.metrics.accuracy(labels=labels,predictions=predictions,name='acc_op')
    #tf.summary.scalar('accuracy', accuracy[1])
    
    if mode == tf.estimator.ModeKeys.EVAL:
    # Compute evaluation metrics.
      #metrics = {'accuracy': accuracy}
      metrics = {}
      return tf.estimator.EstimatorSpec(mode, loss=pred_loss, eval_metric_ops=metrics)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(params['learning_rate'])
        train_op = optimizer.minimize(pred_loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode, loss=pred_loss, train_op=train_op)