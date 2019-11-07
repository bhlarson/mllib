##
# Creates a mobilenet v2 network.
# https://arxiv.org/pdf/1801.04381.pdf
##
def make_mobilenet_v2(self, input_layer, mode):
    is_training = tf.constant(mode == tf.contrib.learn.ModeKeys.TRAIN)
    axis = self.batchnorm_axis()

    def conv_unit(inputs, num_outputs, kernel_size, stride, linear=False):
        print("  conv\t\t\t /s%s \t%s \t%s" % (stride, kernel_size, inputs.get_shape().as_list()))
        prev = tf.contrib.layers.conv2d(inputs=inputs, num_outputs=num_outputs,
                                        kernel_size=kernel_size, stride=stride,
                                        activation_fn=None,
                                        data_format=self.classifier_spec.data_format)
        prev = tf.layers.batch_normalization(inputs=prev, training=is_training, axis=axis, fused=False)
        if not linear:
            prev = tf.nn.relu6(prev)
        return prev

    def depthwise_conv_unit(inputs, num_outputs, stride, depth_multiplier=1):
        print("  depthwise_conv\t /s%s \t\t%s" % (stride, inputs.get_shape().as_list()))
        prev = tf.contrib.layers.separable_conv2d(inputs=inputs, depth_multiplier=depth_multiplier,
                                                  kernel_size=[3, 3], stride=stride,
                                                  num_outputs=None, activation_fn=None,
                                                  data_format=self.classifier_spec.data_format)
        prev = tf.layers.batch_normalization(inputs=prev, training=is_training, axis=axis, fused=False)
        return tf.nn.relu6(prev)

    def bottleneck_unit(inputs, num_outputs, stride, depth_multiplier=1):
        print(" bottleneck filters=%d, stride=%d" % (num_outputs, stride))
        # conv_1x1 relu6 > depthwise_3x3/s1 relu6 > conv 1x1 linear > add_input
        # conv_1x1 relu6 > depthwise_3x3/s2 relu6 > conv_1x1 linear
        prev = conv_unit(inputs, num_outputs=num_outputs, kernel_size=[1, 1], stride=1)
        prev = depthwise_conv_unit(prev, num_outputs=num_outputs, stride=stride, depth_multiplier=depth_multiplier)
        prev = conv_unit(prev, num_outputs=num_outputs, kernel_size=[1, 1], stride=1, linear=True)
        if stride == 1:
            shortcut = conv_unit(inputs, num_outputs=num_outputs, kernel_size=[1, 1], stride=1, linear=True)
            prev = tf.add(shortcut, prev)
        return prev

    def bottleneck_block(inputs, num_outputs, stride, repeats):
        # Each n-sequence has a stride-s unit followed n-1 stride-1 units.
        print("bottleneck filters=%d, stride=%d repeats=%d" % (num_outputs, stride, repeats))
        prev = bottleneck_unit(inputs, num_outputs, stride)
        for i in range(repeats - 1):
            prev = bottleneck_unit(prev, num_outputs, 1)
        return prev

    # Input             Operator        t       c       n       s
    # ==========        ==========     ===     ====    ===     ===
    # 224x224x3         conv2d 3x3      -       32      1       2
    # 112x112x32        bottleneck      1       16      1       1
    # 112x112x16        bottleneck      6       24      2       2
    # 56x56x24          bottleneck      6       32      3       2
    # 28x28x32          bottleneck      6       64      4       2
    # 14x14x64          bottleneck      6       96      3       1
    # 14x14x96          bottleneck      6       160     3       2
    # 7x7x160           bottleneck      6       320     1       1
    # 7x7x320           conv2d 1x1      -       1280    1       1
    # 7x7x1280          avgpool 7x7     -       -       1       -
    # 1x1x1280          conv2d 1x1      -       k       -

    prev = input_layer
    print("input")
    prev = conv_unit(prev, num_outputs=32, kernel_size=[3, 3], stride=2)
    prev = bottleneck_block(prev, num_outputs=16, stride=1, repeats=1)
    prev = bottleneck_block(prev, num_outputs=24, stride=2, repeats=2)
    prev = bottleneck_block(prev, num_outputs=32, stride=2, repeats=3)
    prev = bottleneck_block(prev, num_outputs=64, stride=2, repeats=4)
    prev = bottleneck_block(prev, num_outputs=96, stride=1, repeats=3)
    prev = bottleneck_block(prev, num_outputs=160, stride=2, repeats=3)
    prev = bottleneck_block(prev, num_outputs=320, stride=1, repeats=1)
    prev = conv_unit(prev, num_outputs=1280, kernel_size=[1, 1], stride=1)
    prev = tf.contrib.layers.avg_pool2d(inputs=prev, kernel_size=[7, 7], stride=1)

    print(prev.get_shape().as_list())
    shape = prev.get_shape().as_list()
    prev = tf.reshape(prev, [-1, shape[1] * shape[2] * shape[3]]) # batch size, fc inputs.
    print(prev.get_shape().as_list())
    logits = tf.contrib.layers.fully_connected(inputs=prev, activation_fn=None, num_outputs=len(self.label_map))
    tf.contrib.layers.summarize_activation(logits)
    return logits
