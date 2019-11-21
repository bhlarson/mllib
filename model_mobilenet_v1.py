import tensorflow as tf
from mobilenet_v1 import make_mobilenet_v1
from augment import augment

def loss(logits, labels, num_classes):
    #one_hots = tf.one_hot(labels, num_classes)
    #weights = tf.reduce_sum(one_hots, axis=3)

    max_x_ent_loss = tf.reduce_mean((tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)))
    return max_x_ent_loss

def metrics(labels, predictions, num_classes):

    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_labels)
    precision = tf.metrics.precision(labels=labels, predictions=predicted_labels)
    recall = tf.metrics.recall(labels=labels, predictions=predicted_labels)
    meaniou = tf.metrics.meaniou(labels=labels, predictions=predicted, name='iou_op')

    metrics = {'accuracy': accuracy, 'precision': precision, 'recall':recall , 'meaniou':meaniou}

    return metrics

def default_model_params(args):
    return {'dim':args.dim, 'format':"NHWC"}

def mobilenet_v1_model_fn(features, labels, mode, params = {'dim':[512,768,3], 'format':"NHWC"}):

    if params['format'] == "NHWC":
        iHeight = 0
        iWidth = 1
        batchnorm_axis = 3
    else:
        raise ValueError('mobilenet_v1_model_fn format {} not supported'.format(params['format']))

    def parser(record):
        label_key = "image/label"
        bytes_key = "image/encoded"
        parsed = tf.parse_single_example(record, {
            bytes_key : tf.FixedLenFeature([], tf.string),
            label_key : tf.FixedLenFeature([], tf.int64),
        })
        image = tf.decode_raw(parsed[bytes_key], tf.uint8)
        dims = [width, height, channels]
        image = tf.reshape(image, dims)
        return { "image" : image }, parsed[label_key]                                                            

    images = features["image"]
    images = tf.image.convert_image_dtype(images, tf.float32)
     
    # Test support and performance of NHWC and NHWC
    #if params.data_format == 'NHWC':
        # Convert the inputs from channels_last (NHWC) to channels_first (NHWC).
        # This provides a large performance boost on GPU. See
        # https://www.tensorflow.org/performance/performance_guide#data_formats
        #images = tf.transpose(images, [0, 2, 1, 3], name="nhwc_input")

    # Augment dataset when training
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        images = augment(images, params)

    images = tf.image.per_image_standardization(images)

    images = tf.image.resize_with_crop_or_pad(images,params['dim'][iHeight],params['dim'][iWidth])

    # make_mobilenet_v1
    logits = make_mobilenet_v1(images, mode, batchnorm_axis, params['format'])  

    predictions = {
        # Returns the highest prediction from the output of logits.
        tf.contrib.learn.PredictionKey.CLASSES : tf.argmax(logits, axis=1, name="predicted_classes"),
        # Softmax squashes arbitrary real values into a range of 0 to 1 and returns the
        # probabilities of each of the classes. Name is used for logging.
        tf.contrib.learn.PredictionKey.PROBABILITIES : tf.nn.softmax(logits, name="predicted_probability")
    }

    # Run-time prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        exports = {'predictions' : tf.estimator.export.PredictOutput(predictions)}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=exports)

    #labels = tf.squeeze(labels, axis=3)  # reduce the channel dimension.
    predicted_labels = predictions[tf.contrib.learn.PredictionKey.CLASSES]

    metrics = metrics(labels, predicted_labels, tf.contrib.learn.PredictionKey.CLASSES)  

    tf.summary.scalar("accuracy", metrics['accuracy'][1])
    tf.summary.scalar("precision", metrics['precision'][1])
    tf.summary.scalar("recall", metrics['recall'][1])
    tf.summary.scalar("meaniou", metrics['meaniou'][1])   

    logits_by_num_classes = tf.reshape(logits, [-1, params['num_classes']])
    labels_flat = tf.reshape(labels, [-1, ])
    valid_indices = tf.to_int32(labels_flat <= params['num_classes'] - 1)
    valid_logits = tf.dynamic_partition(logits_by_num_classes, valid_indices, num_partitions=2)[1]
    valid_labels = tf.dynamic_partition(labels_flat, valid_indices, num_partitions=2)[1]

    pred_loss = loss(valid_logits, valid_labels, params['num_classes'])
   
    # Training time evaluation
    if mode == tf.estimator.ModeKeys.EVAL:
      return tf.estimator.EstimatorSpec(mode, loss=pred_loss, eval_metric_ops=metrics)

    # Train the model with AdamOptimizer
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(params['learning_rate'])
        train_op = optimizer.minimize(pred_loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode, loss=pred_loss, train_op=train_op)