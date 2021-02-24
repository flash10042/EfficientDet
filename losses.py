"""EfficientDet losses

This script contains implementations of Focal loss for classification task and
Huber loss for regression task. Also script includes composition of these losses
for quick setup of training pipeline.
"""

import tensorflow as tf


class FocalLoss(tf.keras.losses.Loss):
    """Focal loss implementations."""

    def __init__(self,
                 alpha=0.25,
                 gamma=1.5,
                 label_smoothing=0.1,
                 name='focal_loss'):
        """Initialize parameters for Focal loss.

        FL = - alpha_t * (1 - p_t) ** gamma * log(p_t)
        This implementation also includes label smoothing for preventing overconfidence.
        """
        super().__init__(name=name, reduction="none")
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        """Calculate Focal loss.

        Args:
            y_true: a tensor of ground truth values with
                shape (batch_size, num_anchor_boxes, num_classes).
            y_pred: a tensor of predicted values with
                shape (batch_size, num_anchor_boxes, num_classes).

        Returns:
            A float tensor with shape (batch_size, num_anchor_boxes) with
            loss value for every anchor box.
        """
        prob = tf.sigmoid(y_pred)
        pt = y_true * prob + (1 - y_true) * (1 - prob)
        at = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)

        y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

        loss = at * (1.0 - pt)**self.gamma * ce
        return tf.reduce_sum(loss, axis=-1)


class BoxLoss(tf.keras.losses.Loss):
    """Huber loss implementation."""

    def __init__(self,
                 delta=1.0,
                 name='box_loss'):
        super().__init__(name=name, reduction="none")
        self.delta = delta

    def call(self, y_true, y_pred):
        """Calculate Huber loss.

        Args:
            y_true: a tensor of ground truth values with shape (batch_size, num_anchor_boxes, 4).
            y_pred: a tensor of predicted values with shape (batch_size, num_anchor_boxes, 4).

        Returns:
            A float tensor with shape (batch_size, num_anchor_boxes) with
            loss value for every anchor box.
        """
        loss = tf.abs(y_true - y_pred)
        l1 = self.delta * (loss - 0.5 * self.delta)
        l2 = 0.5 * loss ** 2
        box_loss = tf.where(tf.less(loss, self.delta), l2, l1)
        return tf.reduce_sum(box_loss, axis=-1)


class EffDetLoss(tf.keras.losses.Loss):
    """Composition of Focal and Huber losses."""

    def __init__(self,
                 num_classes=80,
                 alpha=0.25,
                 gamma=1.5,
                 label_smoothing=0.1,
                 delta=1.0,
                 name='effdet_loss'):
        """Initialize Focal and Huber loss.

        Args:
            num_classes: an integer number representing number of
                all possible classes in training dataset.
            alpha: a float number for Focal loss formula.
            gamma: a float number for Focal loss formula.
            label_smoothing: a float number of label smoothing intensity.
            delta: a float number representing a threshold in Huber loss
                for choosing between linear and cubic loss.
        """
        super().__init__(name=name)
        self.class_loss = FocalLoss(alpha=alpha, gamma=gamma, label_smoothing=label_smoothing)
        self.box_loss = BoxLoss(delta=delta)
        self.num_classes = num_classes

    @tf.autograph.experimental.do_not_convert
    def call(self, y_true, y_pred):
        """Calculate Focal and Huber losses for every anchor box.

        Args:
            y_true: a tensor of ground truth values with shape (batch_size, num_anchor_boxes, 5)
                representing anchor box correction and class label.
            y_pred: a tensor of predicted values with
                shape (batch_size, num_anchor_boxes, num_classes).

        Returns:
            loss: a float loss value.
        """
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        box_labels = y_true[..., :4]
        box_preds = y_pred[..., :4]

        cls_labels = tf.one_hot(
            tf.cast(y_true[..., 4], dtype=tf.int32),
            depth=self.num_classes,
            dtype=tf.float32
        )
        cls_preds = y_pred[..., 4:]

        positive_mask = tf.cast(tf.greater(y_true[..., 4], -1.0), dtype=tf.float32)
        ignore_mask = tf.cast(tf.equal(y_true[..., 4], -2.0), dtype=tf.float32)

        clf_loss = self.class_loss(cls_labels, cls_preds)
        box_loss = self.box_loss(box_labels, box_preds)
        clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss)
        box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)

        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer)
        box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)
        loss = clf_loss + box_loss
        return loss
