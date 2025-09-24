import tensorflow as tf
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import keras


def create_mask_dataset(
    img_folder: Path | str,
    mask_folder: Path | str,
    input_shape=(512, 512, 1),
    train: bool = False,
):
    def my_generator(img_folder, mask_folder, input_shape, train=False):
        crop_factors = [1 / 9.0, 1 / 8.0, 1 / 8.0, 1 / 6.0, 1 / 5.0]
        height, width, _channel = input_shape
        if isinstance(img_folder, str):
            img_folder = Path(img_folder)
        if isinstance(mask_folder, str):
            mask_folder = Path(mask_folder)

        for img_file in img_folder.iterdir():
            img = cv2.imread(img_file, flags=cv2.IMREAD_GRAYSCALE)
            img = keras.ops.cast(img, "float32")
            mask_file = mask_folder / img_file.relative_to(img_folder)
            mask = cv2.imread(mask_file, flags=cv2.IMREAD_GRAYSCALE)
            mask = keras.ops.cast(mask, "float32")
            if img.ndim == 2:
                img = keras.ops.expand_dims(img, axis=-1)
            if mask.ndim == 2:
                mask = keras.ops.expand_dims(mask, axis=-1)
            img = keras.ops.image.resize(
                img, size=(height, width), pad_to_aspect_ratio=True
            )
            mask = keras.ops.image.resize(
                mask, size=(height, width), pad_to_aspect_ratio=True
            )
            mask = mask / 255.0
            yield img, mask
            if train:
                img_background = np.zeros_like(img)
                mask_background = np.zeros_like(mask)
                for crop_factor in crop_factors:
                    min_y, max_y = (
                        int(height * crop_factor),
                        int(height * (1 - crop_factor)),
                    )
                    img_background[min_y:max_y, 0:width, :] = img[
                        min_y:max_y, 0:width, :
                    ]
                    mask_background[min_y:max_y, 0:width, :] = mask[
                        min_y:max_y, 0:width, :
                    ]
                    yield img_background, mask_background
                    # cropped_img = keras.ops.image.resize(
                    #     img[min_y:max_y, min_x:max_x, :],
                    #     size=(height, width),
                    #     pad_to_aspect_ratio=True,
                    # )
                    # cropped_mask = keras.ops.image.resize(
                    #     mask[min_y:max_y, min_x:max_x, :],
                    #     size=(height, width),
                    #     pad_to_aspect_ratio=True,
                    # )
                    # yield cropped_img, cropped_mask

    return tf.data.Dataset.from_generator(
        lambda: my_generator(
            img_folder=img_folder,
            mask_folder=mask_folder,
            input_shape=input_shape,
            train=train,
        ),
        output_signature=(
            tf.TensorSpec(shape=input_shape, name="image", dtype="float32"),
            tf.TensorSpec(shape=input_shape, name="mask", dtype="float32"),
        ),
    )


def create_dataset(folder_path: Path | str, num_classes: int, train: bool = False):
    def my_generator(folder_path, augumentation_operations):
        if isinstance(folder_path, str):
            folder_path = Path(folder_path)
        for file in folder_path.iterdir():
            data = np.load(file)
            img, label = data["image"], data["label"]
            if img.ndim < 3:
                img = img.reshape(*img.shape, 1)
            yield img, label
            if train:
                chosen_ops = np.random.choice(
                    augumentation_operations, 2, replace=False
                )
                for op in chosen_ops:
                    img = op(img)
                yield img, label

    operations = [
        lambda img: tf.image.flip_left_right(img),
        lambda img: tf.image.flip_up_down(img),
        lambda img: tf.image.rot90(img),
        lambda img: tf.image.random_brightness(img, max_delta=0.3),
        lambda img: tf.image.random_contrast(img, lower=0.6, upper=1.4),
    ]

    return tf.data.Dataset.from_generator(
        lambda: my_generator(
            folder_path=folder_path, augumentation_operations=operations
        ),
        # output_shapes=((512, 512, 1), (num_classes,)),
        # output_types=(tf.float32, tf.float32),
        output_signature=(
            tf.TensorSpec(shape=(512, 512, 1), name="img", dtype=tf.float32),
            tf.TensorSpec(shape=(num_classes,), name="label", dtype=tf.float32),
        ),
    )


from sklearn.utils import shuffle


def create_images_dataset(
    folder_path: Path | str,
    class_names: list[str],
    train: bool = False,
    target_size=(512, 512),
):
    target_height, target_width = target_size

    def my_generator(folder_path, augumentation_operations, class_names):
        if isinstance(folder_path, str):
            folder_path = Path(folder_path)
        files = [file for file in folder_path.rglob("**/*.jpg")]
        labels = [file.parent.name for file in files]
        files, labels = shuffle(files, labels)
        for file, label in zip(files, labels):
            img = Image.open(file)
            img = np.array(img)
            one_hot = np.zeros(len(class_names), dtype=np.int32)
            one_hot[class_names.index(label)] = 1

            if img.ndim < 3:
                img = img.reshape(*img.shape, 1)
            img = keras.ops.cast(img, "float32") / 255.0
            yield tf.image.resize_with_pad(
                img, target_height=target_height, target_width=target_width
            ), one_hot
            if train:
                for op in augumentation_operations:
                    op_img = op(img)
                    yield tf.image.resize_with_pad(
                        op_img, target_height=target_height, target_width=target_width
                    ), one_hot

    operations = [
        lambda img: tf.image.flip_left_right(img),
        lambda img: tf.image.flip_up_down(img),
        lambda img: tf.image.random_brightness(img, max_delta=0.3),
        lambda img: tf.image.random_contrast(img, lower=0.6, upper=1.4),
    ]

    return tf.data.Dataset.from_generator(
        lambda: my_generator(
            folder_path=folder_path,
            augumentation_operations=operations,
            class_names=class_names,
        ),
        output_signature=(
            tf.TensorSpec(shape=(*target_size, 1), name="img", dtype=tf.float32),
            tf.TensorSpec(shape=(len(class_names),), name="label", dtype=tf.int32),
        ),
    )
