#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

os.environ["KERAS_BACKEND"] = "jax"
import keras

keras.mixed_precision.set_global_policy("mixed_float16")
from layers import UnetBackbone
import numpy as np
import io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

epochs = 300


# Learning rate scheduler
def cosine_annealing_scheduler(epoch, lr):
    initial_lr = 1e-3
    min_lr = 1e-6
    T_max = int(epochs / 2)

    cosine_decay = 0.5 * (1 + np.cos(np.pi * (epoch % T_max) / T_max))
    new_lr = (initial_lr - min_lr) * cosine_decay + min_lr

    return float(new_lr)


import numpy as np
from utils import (
    create_mask_dataset,
    iou_coef,
    dice_coef,
    bfce_dice_loss,
)


# Load datasets using Keras utilities
batch_size = 8

img_size = (1920 // 2, 1920 // 2)
input_shape = (1920 // 2, 1920 // 2, 1)


class PlotMaskCallback(keras.callbacks.Callback):
    def __init__(self, val_ds, log_dir="logs"):
        super().__init__()
        self.file_writer = tf.summary.create_file_writer(f"{log_dir}/masks")
        self.val_ds = val_ds

    def on_epoch_end(self, epoch, logs=None):
        # Log both images to TensorBoard
        with self.file_writer.as_default():
            for batch_index, (images, masks) in enumerate(
                self.val_ds.shuffle(buffer_size=8).take(1)
            ):
                pred_masks = self.model.predict(images)
                tf.summary.image(
                    f"Image - {batch_index}", images / 255.0, step=epoch, max_outputs=4
                )
                tf.summary.image(
                    f"Mask - {batch_index}", masks, step=epoch, max_outputs=4
                )
                tf.summary.image(
                    f"Predicted Mask - {batch_index}",
                    pred_masks,
                    step=epoch,
                    max_outputs=4,
                )


train_ds = create_mask_dataset(
    img_folder="bubble_masks/train/images",
    mask_folder="bubble_masks/train/masks",
    input_shape=input_shape,
    train=True,
)
val_ds = create_mask_dataset(
    img_folder="bubble_masks/val/images",
    mask_folder="bubble_masks/val/masks",
    input_shape=input_shape,
)
train_ds = (
    train_ds.shuffle(buffer_size=100, seed=100)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)
val_ds = val_ds.batch(batch_size)
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="models/unet_semantic_seg_best_loss.keras",
        save_best_only=True,
        monitor="val_loss",
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(epochs / 6)),
    keras.callbacks.TensorBoard(log_dir="logs"),
    # keras.callbacks.LearningRateScheduler(cosine_annealing_scheduler, verbose=1),
    PlotMaskCallback(val_ds=val_ds),
]


# In[ ]:





# In[ ]:





# In[ ]:


# model = keras.Sequential(
#     [
#         keras.Input(input_shape),
#         keras.layers.Rescaling(1.0 / 255),
#         UnetBackbone(general_filters=[32, 64, 128], dropout=0.5),
#         keras.layers.Conv2D(1, kernel_size=3, activation="sigmoid", padding="same"),
#     ]
# )
model = keras.models.load_model('unet_ss_best.keras', compile=False)
model.summary()


# In[ ]:


model.compile(
    loss=bfce_dice_loss,
    optimizer=keras.optimizers.Adam(learning_rate=1e-3, weight_decay=1e-6),
    metrics=[iou_coef, dice_coef],
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks,
)


# In[ ]:




