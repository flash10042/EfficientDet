"""Implementations of layers/models used in EfficientDet."""

import tensorflow as tf
import numpy as np


class BiFPNLayerNode(tf.keras.layers.Layer):
    """One node in BiFPN for features fusing."""

    def __init__(self,
                 channels=64,
                 kernel_size=3,
                 depth_multiplier=1,
                 name='BiFPN_node'):
        """Ininitialize node.

        Args:
            channels: an integer representing number of units inside the node.
            kernel_size: an integer or tuple/list of 2 integers, specifying 
                the height and width of the 2D convolution window.
            depth_multiplier: an integer representing depth multiplier for
                separable convolution layer.
            name: a string representing layer name.
        """
        super().__init__(name=name)
        self.channels = channels
        self.depth_multiplier = depth_multiplier
        self.kernel_size = kernel_size

    def build(self, inputs):
        self.w = self.add_weight(
            shape=(len(inputs), self.channels),
            initializer="ones",
            name='sum_weights',
            trainable=True
        )

        self.conv2d = tf.keras.layers.SeparableConv2D(
            self.channels,
            self.kernel_size,
            padding='same',
            depth_multiplier=self.depth_multiplier,
            pointwise_initializer=tf.initializers.variance_scaling(),
            depthwise_initializer=tf.initializers.variance_scaling(),
            name='node_conv'
        )

        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation(tf.nn.silu)

    def call(self, inputs, training=False):
        """Fuse features.

        Args:
            inputs: a list with length equal to self.w.shape[0] of feature maps
                with equal shapes.

        Returns:
            A float tensor of fused features after applying convolution
            with batch normalization and SiLU activation.
        """
        norm = tf.math.reduce_sum(self.w, axis=0) + 1e-4
        scaled_tensors = [inputs[i] * self.w[i] / norm for i in range(self.w.shape[0])]
        w_sum = tf.math.add_n(scaled_tensors)
        conv = self.conv2d(w_sum)
        bn = self.bn(conv, training=training)
        return self.act(bn)


class BiFPNLayer(tf.keras.layers.Layer):
    """One layer of BiFPN."""

    def __init__(self,
                 channels=64,
                 kernel_size=3,
                 depth_multiplier=1,
                 pooling_strategy='avg',
                 name='BiFPN_Layer'):
        """Initialize BiFPN layer.

        Args:
            channels: an integer representing number of units inside each fusing node.
            kernel_size: an integer or tuple/list of 2 integers, specifying 
                the height and width of the 2D convolution window.
            depth_multiplier: an integer representing depth multiplier for
                separable convolution layers in BiFPN nodes.
            pooling_strategy: a string representing pooling strategy.
                'avg' or 'max'. Otherwise the max pooling will be selected.
            name: a string representing layer name.
        """
        super().__init__(name=name)
        self.pooling_strategy = pooling_strategy

        self.first_step_nodes = [BiFPNLayerNode(channels=channels,
                                                kernel_size=kernel_size,
                                                depth_multiplier=depth_multiplier,
                                                name=f'step_1_level_{i}_node') for i in range(4, 7)]
        self.second_step_nodes = [BiFPNLayerNode(channels=channels,
                                                 kernel_size=kernel_size,
                                                 depth_multiplier=depth_multiplier,
                                                 name=f'step_2_level_{i}_node') for i in range(3, 8)]

    def call(self, inputs, training=False):
        """Perfrom features fusing from different levels."""

        upscaled = self._upscale2d(inputs[-1])
        first_step_outs = [self.first_step_nodes[-1]([inputs[-2], upscaled], training=training)]
        for i in range(2):
            upscaled = self._upscale2d(first_step_outs[i])
            fused = self.first_step_nodes[1-i]([inputs[-3-i], upscaled])
            first_step_outs.append(fused)
 
        upscaled = self._upscale2d(first_step_outs[-1])
        second_step_outs = [self.second_step_nodes[0]([inputs[0], upscaled])]
        for i in range(1, 4):
            downscaled = self._pool2d(second_step_outs[-1])
            fused = self.second_step_nodes[i]([inputs[i], first_step_outs[3-i], downscaled], training=training)
            second_step_outs.append(fused)
        downscaled = self._pool2d(second_step_outs[-1])
        fused = self.second_step_nodes[-1]([inputs[-1], downscaled])
        second_step_outs.append(fused)

        return second_step_outs

    def _pool2d(self, inputs):
        #TODO: Optimize calling of pooling layer.
        if self.pooling_strategy == 'avg':
            return tf.keras.layers.AveragePooling2D()(inputs)
        else:
            return tf.keras.layers.MaxPool2D()(inputs)

    def _upscale2d(self, inputs):
        return tf.keras.layers.UpSampling2D()(inputs)


class BiFPN(tf.keras.layers.Layer):
    """Bidirectional Feature Pyramid Network."""

    def __init__(self,
                 channels=64,
                 depth=3,
                 kernel_size=3,
                 depth_multiplier=1,
                 pooling_strategy='avg',
                 name='BiFPN'):
        super().__init__(name=name)
        """Initialize BiFPN.

        Args:
            channels: an integer representing number of units inside each fusing node
                and convolution layer.
            depth: an integer representing number of BiFPN layers. depth > 0.
            kernel_size: an integer or tuple/list of 2 integers, specifying 
                the height and width of the 2D convolution window.
            depth_multiplier: an integer representing depth multiplier for
                separable convolution layers in BiFPN nodes.
            pooling_strategy: a string representing pooling strategy in BiFPN layers.
                'avg' or 'max'. Otherwise the max pooling will be selected.
            name: a string representing layer name.
        """
        self.depth = depth
        self.channels = channels
        self.pooling_strategy = pooling_strategy

        self.convs_1x1 = [tf.keras.layers.Conv2D(channels,
                                                 1,
                                                 padding='same',
                                                 name=f'1x1_conv_level_{3+i}') for i in range(5)]

        self.bns = [
            tf.keras.layers.BatchNormalization(name=f'bn_level_{i}') for i in range(5)
        ]
        self.act = tf.keras.layers.Activation(tf.nn.silu)

        self.bifpn_layers = [BiFPNLayer(channels=channels,
                                        kernel_size=kernel_size,
                                        depth_multiplier=depth_multiplier,
                                        pooling_strategy=pooling_strategy,
                                        name=f'BiFPN_Layer_{i}') for i in range(depth)]

    def call(self, inputs, training=False):
        assert len(inputs) == 5

        squeezed = [self.convs_1x1[i](inputs[i]) for i in range(5)]
        normalized = [self.bns[i](squeezed[i], training=training) for i in range(5)]
        activated = [self.act(normalized[i]) for i in range(5)]
        feature_maps = self.bifpn_layers[0](activated, training=training)
        for layer in self.bifpn_layers[1:]:
            feature_maps = layer(feature_maps, training=training)

        return feature_maps


class ClassDetector(tf.keras.layers.Layer):
    """Classification head."""

    def __init__(self,
                 num_classes=80,
                 channels=64,
                 num_anchors=9,
                 depth=3,
                 kernel_size=3,
                 depth_multiplier=1,
                 name='class_det'):
        """Initialize classification model.

        Args:
            num_classes: an integer representing number of classes
                to predict.
            channels: an integer representing number of filters
                inside each separable convolution layer.
            num_anchors: an integer representing number of anchor
                boxes.
            depth: an integer representing number of separable
                convolutions before final convolution.
            kernel_size: an integer or tuple/list of 2 integers, specifying 
                the height and width of the 2D convolution window.
            depth_multiplier: an integer representing depth multiplier for
                separable convolution layers.
            name: a string representing layer name.
        """
        super().__init__(name=name)
        self.num_classes = num_classes
        self.channels = channels
        self.depth = depth
        self.kernel_size = kernel_size
        self.depth_multiplier = depth_multiplier

        self.convs = [tf.keras.layers.SeparableConv2D(
            channels,
            kernel_size,
            padding='same',
            depth_multiplier=depth_multiplier,
            pointwise_initializer=tf.initializers.variance_scaling(),
            depthwise_initializer=tf.initializers.variance_scaling(),
            bias_initializer=tf.zeros_initializer(),
            name=f'class_det_separable_conv_{i}'
        ) for i in range(depth)]

        self.bns = [
            tf.keras.layers.BatchNormalization(name=f'bn_{i}') for i in range(depth)
        ]
        self.act = tf.keras.layers.Activation(tf.nn.silu)

        bias_init = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        self.classes = tf.keras.layers.SeparableConv2D(
            num_classes * num_anchors,
            kernel_size,
            padding='same',
            depth_multiplier=depth_multiplier,
            activation=None,
            pointwise_initializer=tf.initializers.variance_scaling(),
            depthwise_initializer=tf.initializers.variance_scaling(),
            bias_initializer=bias_init,
            name='class_preds'
        )

    def call(self, inputs, training=False):
        for i in range(self.depth):
            inputs = self.convs[i](inputs)
            inputs = self.bns[i](inputs, training=training)
            inputs = self.act(inputs)
        class_output = self.classes(inputs)

        return class_output


class BoxRegressor(tf.keras.layers.Layer):
    """Regression head."""

    def __init__(self,
                 channels=64,
                 num_anchors=9,
                 depth=3,
                 kernel_size=3,
                 depth_multiplier=1,
                 name='box_regressor'):
        """Initialize regression model.

        Args:
            channels: an integer representing number of filters
                inside each separable convolution layer.
            num_anchors: an integer representing number of anchor
                boxes.
            depth: an integer representing number of separable
                convolutions before final convolution.
            kernel_size: an integer or tuple/list of 2 integers, specifying
                the height and width of the 2D convolution window.
            depth_multiplier: an integer representing depth multiplier for
                separable convolution layers.
            name: a string representing layer name.
        """
        super().__init__(name=name)
        self.channels=channels
        self.num_anchors=num_anchors
        self.depth=depth
        self.kernel_size=kernel_size
        self.depth_multiplier=depth_multiplier

        self.convs = [tf.keras.layers.SeparableConv2D(
            channels,
            kernel_size,
            padding='same',
            depth_multiplier=depth_multiplier,
            pointwise_initializer=tf.initializers.variance_scaling(),
            depthwise_initializer=tf.initializers.variance_scaling(),
            bias_initializer=tf.zeros_initializer(),
            name=f'box_reg_separable_conv_{i}'
        ) for i in range(depth)]

        self.bns = [
            tf.keras.layers.BatchNormalization(name=f'bn_{i}') for i in range(depth)
        ]
        self.act = tf.keras.layers.Activation(tf.nn.silu)

        self.boxes = tf.keras.layers.SeparableConv2D(
            4 * num_anchors,
            kernel_size,
            padding='same',
            depth_multiplier=depth_multiplier,
            activation=None,
            pointwise_initializer=tf.initializers.variance_scaling(),
            depthwise_initializer=tf.initializers.variance_scaling(),
            bias_initializer=tf.zeros_initializer(),
            name='box_preds'
        )

    def call(self, inputs, training=False):
        for i in range(self.depth):
            inputs = self.convs[i](inputs)
            inputs = self.bns[i](inputs, training=training)
            inputs = self.act(inputs)
        box_output = self.boxes(inputs)

        return box_output
