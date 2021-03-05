"""Inference script."""

import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from model.efficientdet import get_efficientdet
from model.anchors import Anchors
from model.utils import to_corners, resize_and_pad


parser = argparse.ArgumentParser(description='Detect objects on image.')
parser.add_argument('-n', metavar='NAME', default='efficientdet_d0',
                    required=True, help='Name of model to use')
parser.add_argument('-w', metavar='WEIGHTS', required=True, 
                    help='Path to model weights')
parser.add_argument('-i', metavar='IMAGE', required=True, 
                    help='Image to process')
parser.add_argument('-c', metavar='CLASSES', default=80, type=int,
                    help='Number of classes pretrained model predicts.')
parser.add_argument('-a', metavar='ANCHORS', default=9, type=int,
                    help='Number of anchor boxes pretrained model predicts.')
parser.add_argument('-o', metavar='OUTPUT_NAME', default='output.png',
                    help='Name of result image')


def make_prediction(image,
                    max_output_size_per_class=100,
                    max_total_size=100,
                    iot_threshold=0.7,
                    score_threshold=0.1):
    box_variance = tf.cast(
        [0.1, 0.1, 0.2, 0.2], tf.float32
    )
    
    padded_image, new_shape, scale = resize_and_pad(image, scale_jitter=None)
    anchor_boxes = Anchors().get_anchors(padded_image.shape[0], padded_image.shape[1])

    preds = model.predict(tf.expand_dims(padded_image, axis=0))

    boxes = preds[..., :4] * box_variance
    boxes = tf.concat(
        [
            boxes[..., :2] * anchor_boxes[..., 2:] + anchor_boxes[..., :2],
            tf.exp(boxes[..., 2:]) * anchor_boxes[..., 2:]
        ],
        axis=-1
    )
    boxes = to_corners(boxes)
    classes = tf.nn.sigmoid(preds[..., 4:])

    nms = tf.image.combined_non_max_suppression(
        tf.expand_dims(boxes, axis=2),
        classes,
        max_output_size_per_class=max_output_size_per_class,
        max_total_size=max_total_size,
        iou_threshold=iot_threshold,
        score_threshold=score_threshold,
        clip_boxes=False
    )

    valid_dets = nms.valid_detections[0]

    plt.axis('off')
    plt.imshow(image)
    ax = plt.gca()

    for i in range(valid_dets):
        x_min, y_min, x_max, y_max = nms.nmsed_boxes[0, i] / scale
        w, h = x_max - x_min, y_max - y_min
        x_min, y_min, w, h = 75, 40, 35, 20
        patch = plt.Rectangle(
            [x_min, y_min], w, h, fill=False, edgecolor=[0, 1, 0], linewidth=1
        )
        ax.add_patch(patch)
        ax.text(
            x_min, y_min, f'Class {int(nms.nmsed_classes[0, i])}: {nms.nmsed_scores[0, i]}',
            bbox={"facecolor": [0, 1, 0], "alpha": 0.4}, clip_box=ax.clipbox,
            clip_on=True
        )

    plt.savefig(args.o)


args = parser.parse_args()

model = get_efficientdet(args.n, num_classes=args.c, num_anchors=args.a)
model.load_weights(args.w)

raw_image = tf.io.read_file(args.i)
image = tf.image.decode_image(raw_image, channels=3)

make_prediction(image)
