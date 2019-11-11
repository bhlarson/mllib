from mobilenet_v1 import make_mobilenet_v1
from augmentations.py import augment

def mobilenet_v1_model_fn(features, labels, mode, params = {height = 512, width=512,data_format="NHWC"}):

    images_uint8 = features["image"]
    
    # Convert the range of the pixels from 0 - 255 to 0.0 - 1.0
    image = tf.image.convert_image_dtype(images_uint8, tf.float32)

    if data_format == 'NHWC':
        # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
        # This provides a large performance boost on GPU. See
        # https://www.tensorflow.org/performance/performance_guide#data_formats
        image = tf.transpose(image, [0, 3, 1, 2])

    # Augment dataset when training
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        image = augment(image)

    image = tf.image.resize_with_crop_or_pad(image,params.height,params.width)

    make_mobilenet_v1
    logits = make_mobilenet_v1(image, mode)  

    #predictions = {
    #    'class_ids': pred,
        #'probabilities': tf.nn.softmax(logits),
    #    'logits': logits,
    #}

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