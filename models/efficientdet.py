"""Implementation of EffDets."""

import tensorflow as tf
from .layers import BiFPN, ClassDetector, BoxRegressor


class EfficientDetD0(tf.keras.Model):
    def __init__(self,
                 channels=64,
                 num_classes=80,
                 name='efficientdet'):
        super().__init__(name=name)
        self.num_classes=num_classes

        self.backbone = efficientnetb0_backbone()
        self.backbone.trainable = False

        self.BiFPN = BiFPN(channels=channels)
        self.class_det = ClassDetector(channels=channels, num_classes=num_classes)
        self.box_reg = BoxRegressor(channels=channels)

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]

        features = self.backbone(inputs, training=training)
        features.append(tf.keras.layers.AveragePooling2D()(features[-1]))
        features.append(tf.keras.layers.AveragePooling2D()(features[-1]))

        fpn_features = self.BiFPN(features)

        classes = list()
        boxes = list()
        for feature in fpn_features:
            classes.append(tf.reshape(self.class_det(feature), [batch_size, -1, self.num_classes]))
            boxes.append(tf.reshape(self.box_reg(feature), [batch_size, -1, 4]))

        classes = tf.concat(classes, axis=1)
        boxes = tf.concat(boxes, axis=1)

        return tf.concat([boxes, classes], axis=-1)


#TODO CREATE MODEL FACTORY FOR DIFFERENT SCALES OF MODEL

ENDPOINTS = [
    'block3b_add',
    'block5c_add',
    'top_activation' # OR TRY top_bn
]

def efficientnetb0_backbone(name='eff_net_b0', 
                            weights='imagenet'):
    backbone = tf.keras.applications.EfficientNetB0(include_top=False, 
                                                    weights=weights,
                                                    input_shape=[None, None, 3])
    outputs = [backbone.get_layer(layer_name).output for layer_name in ENDPOINTS]
    return tf.keras.Model(inputs=backbone.inputs, outputs=outputs, name=name)
