"""Script for creating and training a new model."""

import tensorflow as tf
from model.efficientdet import get_efficientdet
from model.losses import EffDetLoss
from model.anchors import SamplesEncoder

MODEL_NAME = 'efficientdet_d0'

NUM_CLASSES = 80

EPOCHS = 300
BATCH_SIZE = 4

INITIAL_LR = 0.01
DECAY_STEPS = 433 * 155
LR = tf.keras.experimental.CosineDecay(init_lr, DECAY_STEPS, 1e-3)

CHECKPOINT_PATH = '/path/to/checkpoints/folder'

# TODO: LOAD YOUR TRAINING DATA
# TRAINING DATA SHOUD BE IN FORMAT (Image, Bounding boxes, Class labels)
train_data = '/path/to/training/data'

samples_encoder = SamplesEncoder()
autotune = tf.data.experimental.AUTOTUNE

train_data = train_data.shuffle(5000)
train_data = train_data.padded_batch(BATCH_SIZE, padding_values=(0.0, 1e-8, -1.0))
train_data = train_data.map(samples_encoder.encode_batch, num_parallel_calls=autotune)
train_data = train_data.prefetch(autotune)

model = get_efficientdet(MODEL_NAME, num_classes=NUM_CLASSES)
loss = EffDetLoss(num_classes=NUM_CLASSES)
opt = tf.keras.optimizers.SGD(LR, momentum=0.9)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_weights_only=True)

model.fit(train_data, epochs=EPOCHS, callbacks=[checkpoint_callback]. use_multiprocessing=True)
