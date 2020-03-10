import tensorflow as tf

##
# Makes the mobilenet v1 architecture.
# Mobilenet: https://arxiv.org/pdf/1704.04861.pdf
##
def make_mobilenet_v1(input_layer, mode, axis, data_format, num_classes):
        is_training = tf.constant(mode == tf.contrib.learn.ModeKeys.TRAIN)

        def conv_unit(inputs, num_outputs, kernel_size, stride):
            # Conv layers are 3x3 > BatchNorm > ReLu
            print("conv\t\t /s%s \t%s \t%s" % (stride, kernel_size, inputs.get_shape().as_list()))
            prev =  tf.contrib.layers.conv2d(inputs=inputs, num_outputs=num_outputs,
                                             kernel_size=kernel_size, stride=stride,
                                             activation_fn=None, data_format=data_format)
            prev = tf.layers.batch_normalization(inputs=prev, training=is_training, axis=axis, fused=False)
            return tf.nn.relu(prev)

        def depthwise_conv_unit(inputs, num_outputs, stride, depth_multiplier=1):
            # Depthwise conv layers are 3x3 depthwise > BatchNorm > ReLu > 1x1 Conv > BatchNorm > ReLu
            print("depthwise_conv\t /s%s \t\t%s" % (stride, inputs.get_shape().as_list()))
            prev = tf.contrib.layers.separable_conv2d(inputs=inputs, depth_multiplier=depth_multiplier, kernel_size=[3, 3],
                                                      stride=stride, num_outputs=None, activation_fn=None,
                                                      data_format=data_format)
            prev = tf.layers.batch_normalization(inputs=prev, training=is_training, axis=axis, fused=False)
            prev = tf.nn.relu(prev)
            return conv_unit(inputs=prev, num_outputs=num_outputs, kernel_size=[1, 1], stride=1)

        prev = input_layer
        # tf.nn.relu is the default activation function.
        #   Type / Stride         Filter Shape          Input Size
        #   =============       =================       ==============
        #   Conv / s2           3 × 3 × 3 × 32          224 × 224 × 3  #1
        #   Conv dw / s1        3 × 3 × 32 dw           112 × 112 × 32 #2
        #   Conv / s1           1 × 1 × 32 × 64         112 × 112 × 32
        #   Conv dw / s2        3 × 3 × 64 dw           112 × 112 × 64 #3
        #   Conv / s1           1 × 1 × 64 × 128        56 × 56 × 64
        #   Conv dw / s1        3 × 3 × 128 dw          56 × 56 × 128  #4
        #   Conv / s1           1 × 1 × 128 × 128       56 × 56 × 128
        #   Conv dw / s2        3 × 3 × 128 dw          56 × 56 × 128  #5
        #   Conv / s1           1 × 1 × 128 × 256       28 × 28 × 128
        #   Conv dw / s1        3 × 3 × 256 dw          28 × 28 × 256  #6
        #   Conv / s1           1 × 1 × 256 × 256       28 × 28 × 256
        #   Conv dw / s2        3 × 3 × 256 dw          28 × 28 × 256  #7
        #   Conv / s1           1 × 1 × 256 × 512       14 × 14 × 256
        #5× ---------------------------------------------------------- #8-12
        #   Conv dw / s1        3 × 3 × 512 dw          14 × 14 × 512
        #   Conv / s1           1 × 1 × 512 × 512       14 × 14 × 512
        #   ---------------------------------------------------------
        #
        #   Conv dw / s2        3 × 3 × 512 dw          14 × 14 × 512  #13
        #   Conv / s1           1 × 1 × 512 × 1024      7 × 7 × 512
        #   Conv dw / s1        3 × 3 × 1024 dw         7 × 7 × 1024   #14
        #   Conv / s1           1 × 1 × 1024 × 1024     7 × 7 × 1024
        #
        #   Avg Pool / s1       Pool 7 × 7              7 × 7 × 1024
        #   FC / s1             1024 × 1000             1 × 1 × 1024
        #   Softmax / s1        Classifier              1 × 1 × 1000
        prev = conv_unit(inputs=prev, num_outputs=32, kernel_size=[3, 3], stride=2) #1
        prev = depthwise_conv_unit(inputs=prev, num_outputs=64, stride=1)           #2
        prev = depthwise_conv_unit(inputs=prev, num_outputs=128, stride=2)          #3
        prev = depthwise_conv_unit(inputs=prev, num_outputs=128, stride=1)          #4
        prev = depthwise_conv_unit(inputs=prev, num_outputs=256, stride=2)          #5
        prev = depthwise_conv_unit(inputs=prev, num_outputs=256, stride=1)          #6
        prev = depthwise_conv_unit(inputs=prev, num_outputs=512, stride=2)          #7
        print("--")
        for i in range(0, 5):
            prev = depthwise_conv_unit(inputs=prev, num_outputs=512, stride=1)      #8-12
        print("--")
        prev = depthwise_conv_unit(inputs=prev, num_outputs=1024, stride=2)         #13
        prev = depthwise_conv_unit(inputs=prev, num_outputs=1024, stride=1)         #14

        prev = tf.contrib.layers.avg_pool2d(inputs=prev, kernel_size=[7, 7], stride=1)
        print(prev.get_shape().as_list())
        shape = prev.get_shape().as_list()
        prev = tf.reshape(prev, [-1, shape[1] * shape[2] * shape[3]]) # batch size, fc inputs.
        print(prev.get_shape().as_list())
        prev = tf.contrib.layers.fully_connected(inputs=prev, num_outputs=1024)
        print(prev.get_shape().as_list())
        logits = tf.contrib.layers.fully_connected(inputs=prev, activation_fn=None, num_outputs=num_classes)
        tf.contrib.layers.summarize_activation(logits)

        return logits
