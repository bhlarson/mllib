import tensorflow as tf
    
##
# Return the channels axis based on the data format.
##
def batchnorm_axis(classifier_spec):
    if classifier_spec.data_format == "NHWC":
        return 3
    return 1

##
# Creates a resnet50 network.
# https://arxiv.org/pdf/1512.03385.pdf
##
def make_resnet50(classifier_spec, input_layer, mode):
        data_format = classifier_spec.data_format
        is_training = tf.constant(mode == tf.contrib.learn.ModeKeys.TRAIN)
        axis = batchnorm_axis(classifier_spec)

        # Fixed padding adds in padding based only on the kernel size. This is how cudnn and caffe
        # do it. Tensorflow has a different padding scheme that involves the size of the input.
        def pad(input_layer, kernel_size):
            pad_total = kernel_size - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            if classifier_spec.data_format == "NHWC":
                padded = tf.pad(input_layer, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
            else:
                padded = tf.pad(input_layer, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
            return padded

        # Conv2d with fancy resnet padding.
        def conv2d(input_layer, filters, kernel_size, stride):
            print("    conv2d input=%s filters=%d, kernel_size=%d stride=%d"
                    % (input_layer.get_shape().as_list(), filters, kernel_size, stride))
            prev = input_layer
            if (stride > 1):
                prev = pad(input_layer, kernel_size)
            prev = tf.contrib.layers.conv2d(inputs=prev, num_outputs=filters,
                    kernel_size=kernel_size, stride=stride,
                    padding=('SAME' if stride == 1 else 'VALID'),
                    activation_fn=None, data_format=data_format)
            return prev

        # A single bottleneck block of resnet50.
        def bottleneck_block(input_layer, filters, stride, projection_shortcut):
            # A single resnet50 bottleneck block looks like this:
            #
            #            |
            #            +------------------------+
            #            |                        |
            #    1x1 conv f filters               |
            #        batch norm                   |
            #          relu                       |
            #            |                        |
            #    3x3 conv f filters               |
            #        batch norm          [1x1 conv, 4f filters] <- projection shortcut
            #          relu                  [batch norm]
            #            |                        |
            #    1x1 conv 4f filters              |
            #        batch norm                   |
            #          relu                       |
            #            |                        |
            #           add-----------------------+
            #            |
            #          relu
            #            |
            prev = input_layer

            if projection_shortcut:
                shortcut = conv2d(input_layer, filters=(filters * 4), kernel_size=1, stride=stride)
                shortcut = tf.layers.batch_normalization(inputs=shortcut, training=is_training,
                                                         axis=axis, fused=False)
            else:
                shortcut = input_layer

            prev = conv2d(input_layer=prev, filters=filters, kernel_size=1, stride=1)
            prev = tf.layers.batch_normalization(inputs=prev, training=is_training,
                                                 axis=axis, fused=False)
            prev = tf.nn.relu(prev)

            prev = conv2d(input_layer=prev, filters=filters, kernel_size=3, stride=stride)
            prev = tf.layers.batch_normalization(inputs=prev, training=is_training,
                                                 axis=axis, fused=False)
            prev = tf.nn.relu(prev)

            prev = conv2d(input_layer=prev, filters=(4 * filters), kernel_size=1, stride=1)
            prev = tf.layers.batch_normalization(inputs=prev, training=is_training,
                                                 axis=axis, fused=False)
            prev = tf.nn.relu(prev)

            prev = tf.add(prev, shortcut)
            return tf.nn.relu(prev)

        # A group of bottlneck blocks.
        def resnet_block(input_layer, filters, stride, num_bottlenecks):
            print("  resnet block input=%s filters=%d, stride=%d, num_bottlenecks=%d"
                    % (input_layer.get_shape().as_list(), filters, stride, num_bottlenecks))
            prev = bottleneck_block(input_layer, filters=filters, stride=stride, projection_shortcut=True)
            for i in range(1, num_bottlenecks):
                prev = bottleneck_block(prev, filters=filters, stride=1, projection_shortcut=False)
            return prev

        # Resnet50 has the following blocks.
        # conv 7x7x64 / s2
        # batch norm
        # relu
        # maxpool 3x3 / s2
        # -------------------------
        # 3 x  64 filters, stride 1 [Block 1]
        # 4 x 128 filters, stride 2 [Block 2]
        # 6 x 256 filters, stride 2 [Block 3]
        # 3 x 512 filters, stride 2 [Block 4]
        # -------------------------
        # average pool 7x7 stride 1
        # fc 2048
        prev = conv2d(input_layer, filters=64, kernel_size=7, stride=2)
        prev = tf.layers.batch_normalization(inputs=prev, training=is_training, axis=axis, fused=False)
        prev = tf.nn.relu(prev)
        prev = tf.contrib.layers.max_pool2d(inputs=prev, kernel_size=3, stride=2, padding='SAME')

        print("Block 1 input=%s" % prev.get_shape().as_list())
        prev = resnet_block(prev, filters=64,  stride=1, num_bottlenecks=3)
        print("Block 2 input=%s" % prev.get_shape().as_list())
        prev = resnet_block(prev, filters=128, stride=2, num_bottlenecks=4)
        print("Block 3 input=%s" % prev.get_shape().as_list())
        prev = resnet_block(prev, filters=256, stride=2, num_bottlenecks=6)
        print("Block 4 input=%s" % prev.get_shape().as_list())
        prev = resnet_block(prev, filters=512, stride=2, num_bottlenecks=3)

        print("avg pool input=%s" % prev.get_shape().as_list())
        prev = tf.contrib.layers.avg_pool2d(inputs=prev, kernel_size=[7, 7], stride=1)

        shape = prev.get_shape().as_list()
        prev = tf.reshape(prev, [-1, shape[1] * shape[2] * shape[3]]) # batch size, fc inputs.
        print("dense input=%s" % prev.get_shape().as_list())
        logits = tf.contrib.layers.fully_connected(inputs=prev, activation_fn=None, num_outputs=len(self.label_map))

        tf.contrib.layers.summarize_activation(logits)
        return logits
