"""Implementation of utility functions."""

import tensorflow as tf


@tf.autograph.experimental.do_not_convert
def to_xywh(bbox):
    """Convert [x_min, y_min, x_max, y_max] to [x, y, width, height]."""
    return tf.concat(
        [(bbox[..., :2] + bbox[..., 2:]) / 2.0, (bbox[..., 2:] - bbox[..., :2])], axis=-1
    )


@tf.autograph.experimental.do_not_convert
def to_corners(bbox):
    """Convert [x, y, width, height] to [x_min, y_min, x_max, y_max]."""
    return tf.concat(
        [bbox[..., :2] - bbox[..., 2:] / 2.0, bbox[..., :2] + bbox[..., 2:] / 2.0], axis=-1
    )


@tf.autograph.experimental.do_not_convert
def compute_iou(boxes_1, boxes_2):
    """Compute intersection over union.

    Args:
        boxes_1: a tensor with shape (N, 4) representing bounding boxes
            where each box is of the format [x, y, width, height].
        boxes_2: a tensor with shape (M, 4) representing bounding boxes
            where each box is of the format [x, y, width, height].

    Returns:
        IOU matrix with shape (N, M).
    """

    boxes_1_corners = to_corners(boxes_1)
    boxes_2_corners = to_corners(boxes_2)

    left_upper = tf.maximum(boxes_1_corners[..., None, :2], boxes_2_corners[..., :2])
    right_lower = tf.minimum(boxes_1_corners[..., None, 2:], boxes_2_corners[..., 2:])
    diff = tf.maximum(0.0, right_lower - left_upper)
    intersection = diff[..., 0] * diff[..., 1]

    boxes_1_area = boxes_1[..., 2] * boxes_1[..., 3]
    boxes_2_area = boxes_2[..., 2] * boxes_2[..., 3]
    union = boxes_1_area[..., None] + boxes_2_area - intersection

    iou = intersection / union
    return tf.clip_by_value(iou, 0.0, 1.0)


def random_horizontal_flip(image, boxes):
    """Flip image and boxes horizontally."""

    if tf.random.uniform(()) >= 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack(
            [1 - boxes[..., 2], boxes[..., 1], 1 - boxes[..., 0], boxes[..., 3]], axis=-1
        )
    return image, boxes


def resize_and_pad(image, target_side=512.0, max_side=1024.0, scale_jitter=[0.1, 2.0], stride=128.0):
    """Resize image, apply scale jittering and pad with zeros to make image divisible by stride."""

    image_shape = tf.cast(tf.shape(image)[:2], tf.float32)
    bigger_side = tf.reduce_max(image_shape)
    target_side = target_side
    if scale_jitter:
        target_side = bigger_side * tf.random.uniform((), scale_jitter[0], scale_jitter[1], dtype=tf.float32)
    scale_coeff = target_side / bigger_side
    if target_side > max_side:
        scale_coeff = max_side / bigger_side

    new_image_shape = image_shape * scale_coeff
    new_image = tf.image.resize(image, tf.cast(new_image_shape, tf.int32))
    padded_image_shape = tf.cast(tf.math.ceil(new_image_shape / stride) * stride, dtype=tf.int32)
    padded_image = tf.image.pad_to_bounding_box(
        new_image, 0, 0, padded_image_shape[0], padded_image_shape[1]
    )

    return padded_image, new_image_shape, scale_coeff
