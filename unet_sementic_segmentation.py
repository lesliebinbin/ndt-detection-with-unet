#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

os.environ["KERAS_BACKEND"] = "jax"
import keras

# keras.mixed_precision.set_global_policy("mixed_float16")
from layers import UnetBackbone, unet_model
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
    create_replicated_img_dataset,
    create_mask_dataset,
    create_mask_slices_dataset,
    iou_coef,
    dice_coef,
    bfce_dice_loss,
)


# Load datasets using Keras utilities
batch_size = 16

img_size = (512, 512)
input_shape = (512, 512, 1)


class PlotMaskCallback(keras.callbacks.Callback):
    def __init__(self, val_ds, log_dir="logs", max_output=4):
        super().__init__()
        self.file_writer = tf.summary.create_file_writer(f"{log_dir}/masks")
        self.val_ds = val_ds
        self.max_output = max_output

    def on_epoch_end(self, epoch, logs=None):
        # Log both images to TensorBoard
        with self.file_writer.as_default():
            for images, masks in (
                self.val_ds.batch(self.max_output)
                .shuffle(buffer_size=self.max_output * 2)
                .take(1)
            ):
                pred_masks = self.model.predict(images)
                tf.summary.image(
                    "Image",
                    images / 255.0,
                    step=epoch,
                    max_outputs=self.max_output,
                )
                tf.summary.image(
                    "Mask",
                    masks,
                    step=epoch,
                    max_outputs=self.max_output,
                )
                tf.summary.image(
                    "Predicted Mask",
                    pred_masks,
                    step=epoch,
                    max_outputs=self.max_output,
                )


train_ds = create_mask_dataset(
    img_folder="undercut_obb/train/images",
    mask_folder="undercut_obb/train/masks",
    input_shape=input_shape,
    train=True,
    # transforms=[
    #     keras.layers.RandomRotation(0.25),
    #     keras.layers.RandomErasing(fill_value=0),
    # ],
)
origin_val_ds = create_mask_dataset(
    img_folder="undercut_obb/valid/images",
    mask_folder="undercut_obb/valid/masks",
    input_shape=input_shape,
)
train_ds = train_ds.batch(batch_size)
val_ds = origin_val_ds.batch(batch_size)
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="models/unet_semantic_seg_best_loss.keras",
        save_best_only=True,
        monitor="val_loss",
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=100),
    keras.callbacks.TensorBoard(log_dir="logs"),
    # keras.callbacks.LearningRateScheduler(cosine_annealing_scheduler, verbose=1),
    PlotMaskCallback(val_ds=origin_val_ds, log_dir="logs", max_output=8),
]


# In[ ]:


model = keras.Sequential(
    [
        keras.Input(input_shape),
        keras.layers.Rescaling(1.0 / 255),
        unet_model(
            input_shape=input_shape,
            depth=4,
            initial_filter=32,
            use_batch_norm=True,
        ),
    ]
)
# model = keras.models.load_model('unet_ss_undercut.keras', compile=False)
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




