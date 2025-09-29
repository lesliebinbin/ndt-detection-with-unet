import tensorflow as tf
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import keras
from functools import partial


def _custom_crop_image(
    origin_img,
    origin_mask,
    crop_x_min,
    crop_x_max,
    crop_y_min,
    crop_y_max,
):
    img_background = np.zeros_like(origin_img)
    mask_background = np.zeros_like(origin_mask)
    img_background[crop_y_min:crop_y_max, crop_x_min:crop_x_max, :] = origin_img[
        crop_y_min:crop_y_max, crop_x_min:crop_x_max, :
    ]
    mask_background[crop_y_min:crop_y_max, crop_x_min:crop_x_max, :] = origin_mask[
        crop_y_min:crop_y_max, crop_x_min:crop_x_max, :
    ]
    return img_background, mask_background


def _preprocess_image_and_mask(img, mask, target_size=None, transform=None):
    if transform is not None:
        _transform = transform.get_random_transformation(img)
        transformed_img = transform.transform_images(img, _transform)
        transformed_mask = transform.transform_images(mask, _transform)
        if transformed_img.ndim > img.ndim:
            transformed_img = transformed_img[0]
        if transformed_mask.ndim > mask.ndim:
            transformed_mask = transformed_mask[0]
        img, mask = transformed_img, transformed_mask
    if mask.max() > 1.0:
        mask = mask / 255.0
    if target_size is not None:
        img, mask = keras.ops.image.resize(
            keras.ops.array([img, mask]), size=target_size, pad_to_aspect_ratio=True
            keras.ops.array([img, mask]), size=target_size, pad_to_aspect_ratio=True
        )
    return img, mask


def _crop_coordinates(mask):
    img_height, img_width = mask.shape[:2]
    y_coords, _x_coords, _z_coords = keras.ops.where(mask > 0)
    if len(y_coords) > 0:
        min_y, max_y = (
            np.min(y_coords),
            np.max(y_coords),
        )
        crop_x_min = 0
        crop_x_max = img_width
        crop_y_min = 0 if min_y == 0 else np.random.choice(range(0, min_y))
        crop_y_max = (
            img_height
            if max_y + 1 == img_height
            else np.random.choice(range(max_y + 1, img_height))
        )
        return crop_x_min, crop_x_max, crop_y_min, crop_y_max
    return None


def _random_crop_coordinates(mask, preserve_height=False, preserve_width=False):
    height, width = mask.shape[:2]
    if preserve_height:
        crop_height = height
    else:
        crop_height = np.random.choice(range(height // 2, height))
    if preserve_width:
        crop_width = width
    else:
        crop_width = np.random.choice(range(width // 2, width))
    crop_center_y = height // 2
    crop_center_x = width // 2
    crop_x_min = crop_center_x - crop_width // 2
    crop_x_max = crop_center_x + crop_width // 2
    crop_y_min = crop_center_y - crop_height // 2
    crop_y_max = crop_center_y + crop_height // 2
    return crop_x_min, crop_x_max, crop_y_min, crop_y_max


def _resize_and_pad_to_aspect_ratio(image_and_masks, target_height, target_width):
    images_and_masks_arr = keras.ops.array(image_and_masks)
    resized_images_and_masks = keras.ops.image.resize(
        images_and_masks_arr,
        size=(target_height, target_width),
        pad_to_aspect_ratio=True,
    )
    reshaped_images_and_masks = keras.ops.reshape(
        resized_images_and_masks, (-1, 2, *resized_images_and_masks.shape[1:])
    )

    return [
        (image, mask)
        for image, mask in zip(
            reshaped_images_and_masks[:, 0, :, :, :],
            reshaped_images_and_masks[:, 1, :, :, :],
        )
    ]


def _replicate_img_and_mask(img, mask, target_shape=None):
    height, width = img.shape
    if height > width:
        img = img.T
        mask = mask.T
        height, width = width, height
    img_background = np.zeros(shape=(width, width), dtype=img.dtype)
    mask_background = np.zeros(shape=(width, width), dtype=mask.dtype)
    start_index = 0
    while start_index + height < width:
        img_background[start_index : start_index + height, :] = img
        mask_background[start_index : start_index + height, :] = mask
        start_index += height
    if start_index < width:
        img_background[start_index:width, :] = img[0 : (width - start_index), :]
        mask_background[start_index:width, :] = mask[0 : (width - start_index), :]
    img_background = keras.ops.expand_dims(img_background, axis=-1)
    mask_background = keras.ops.expand_dims(mask_background, axis=-1)
    replicated_img = keras.ops.cast(img_background, "float32")
    replicated_mask = keras.ops.cast(mask_background, "float32") / 255.0
    if target_shape is not None:
        replicated_img, replicated_mask = keras.ops.image.resize(
            keras.ops.array([replicated_img, replicated_mask]), size=target_shape
        )
    return replicated_img, replicated_mask


def create_replicated_img_dataset(
    img_folder: Path | str,
    mask_folder: Path | str,
    input_shape=(512, 512, 1),
):
    def my_generator(img_folder, mask_folder, input_shape):
        height, width, _channel = input_shape
        if isinstance(img_folder, str):
            img_folder = Path(img_folder)
        if isinstance(mask_folder, str):
            mask_folder = Path(mask_folder)
        for img_file in img_folder.iterdir():
            mask_file = mask_folder / img_file.relative_to(img_folder)
            img = cv2.imread(img_file, flags=cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_file, flags=cv2.IMREAD_GRAYSCALE)
            yield _replicate_img_and_mask(img, mask, (height, width))

    return tf.data.Dataset.from_generator(
        lambda: my_generator(
            img_folder=img_folder,
            mask_folder=mask_folder,
            input_shape=input_shape,
        ),
        output_signature=(
            tf.TensorSpec(shape=input_shape, name="image", dtype="float32"),
            tf.TensorSpec(shape=input_shape, name="mask", dtype="float32"),
        ),
    )


def calculate_coordinates(height, width, target_width, step, random_walk=False):
    coordinates = []
    if not random_walk:
        start_width = 0
        while start_width + target_width < width:
            min_x, min_y, max_x, max_y = (
                start_width,
                0,
                start_width + target_width,
                height,
            )
            coordinates.append([min_x, min_y, max_x, max_y])
            start_width += step
        coordinates.append([width - target_width, 0, width, height])
    else:
        start_width, end_width = (
            height // 2,
            width - height // 2,
        )
        width_centers = np.random.choice(
            range(start_width, end_width),
            replace=False,
            size=(width - height) // step,
        )
        for width_center in width_centers:
            min_x, min_y, max_x, max_y = (
                width_center - height // 2,
                0,
                width_center + height // 2,
                height,
            )
            coordinates.append([min_x, min_y, max_x, max_y])
    return coordinates


def img_and_mask_slice(img, mask, coordinates):
    if img.ndim == 2:
        img = keras.ops.expand_dims(img, axis=-1)
    if mask.ndim == 2:
        mask = keras.ops.expand_dims(mask, axis=-1)

    return [
        (img[min_y:max_y, min_x:max_x, :], mask[min_y:max_y, min_x:max_x, :])
        for (min_x, min_y, max_x, max_y) in coordinates
    ]


def create_mask_slices_dataset(
    img_folder: Path | str,
    mask_folder: Path | str,
    input_shape=(512, 512, 1),
    step_factor=0.25,
    train: bool = False,
):

    def my_generator(img_folder, mask_folder, input_shape, step_factor, train=False):
        target_height, target_width, _channel = input_shape
        if isinstance(img_folder, str):
            img_folder = Path(img_folder)
        if isinstance(mask_folder, str):
            mask_folder = Path(mask_folder)
        for img_file in img_folder.iterdir():
            coordinates = []
            mask_file = mask_folder / img_file.relative_to(img_folder)
            img = cv2.imread(img_file, flags=cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_file, flags=cv2.IMREAD_GRAYSCALE)
            height, width = img.shape
            if height > width:
                img = img.T
                mask = mask.T
                height, width = width, height
            resized_height = target_height
            scale_factor = resized_height / height
            resized_width = int(width * scale_factor)
            img = cv2.resize(img, dsize=(resized_width, resized_height)).astype(
                "float32"
            )
            mask = (
                cv2.resize(mask, dsize=(resized_width, resized_height)).astype(
                    "float32"
                )
                / 255.0
            )
            step = int(step_factor * resized_height)
            if train:
                coordinates.extend(
                    calculate_coordinates(
                        resized_height,
                        resized_width,
                        target_width,
                        step=step,
                        random_walk=True,
                    )
                )
            coordinates.extend(
                calculate_coordinates(
                    resized_height, resized_width, target_width, step=step
                )
            )

            yield from img_and_mask_slice(img, mask, coordinates)

    return tf.data.Dataset.from_generator(
        lambda: my_generator(
            img_folder=img_folder,
            mask_folder=mask_folder,
            input_shape=input_shape,
            step_factor=step_factor,
            train=train,
        ),
        output_signature=(
            tf.TensorSpec(shape=input_shape, name="image", dtype="float32"),
            tf.TensorSpec(shape=input_shape, name="mask", dtype="float32"),
        ),
    )


def create_mask_dataset(
    img_folder: Path | str,
    mask_folder: Path | str,
    input_shape=(512, 512, 1),
    train: bool = False,
    transforms=None,
    transforms=None,
):
    if isinstance(img_folder, str):
        img_folder = Path(img_folder)
    if isinstance(mask_folder, str):
        mask_folder = Path(mask_folder)

    def my_generator():
        input_height, input_width, _channel = input_shape

        for img_file in img_folder.iterdir():
            mask_file = mask_folder / img_file.relative_to(img_folder)
            origin_img, origin_mask = cv2.imread(
                img_file, flags=cv2.IMREAD_GRAYSCALE
            ).astype("float32"), cv2.imread(
                mask_file, flags=cv2.IMREAD_GRAYSCALE
            ).astype(
                "float32"
            )
            origin_img = keras.ops.expand_dims(origin_img, axis=-1)
            origin_mask = keras.ops.expand_dims(origin_mask, axis=-1)
            yield _preprocess_image_and_mask(
                origin_img, origin_mask, target_size=(input_height, input_width)
            )

            if train and transforms:
                for transform in transforms:
                    yield _preprocess_image_and_mask(
                        origin_img,
                        origin_mask,
                        target_size=(input_height, input_width),
                        transform=transform,
                    )
                    extra_to_yield = [
                        (
                            origin_img[:, ::-1, :],
                            origin_mask[:, ::-1, :],
                        ),
                        (
                            origin_img[::-1, :, :],
                            origin_mask[::-1, :, :],
                        ),
                        (
                            origin_img[::-1, ::-1, :],
                            origin_mask[::-1, ::-1, :],
                        ),
                    ]
                    for extra_img, extra_mask in extra_to_yield:
                        yield _preprocess_image_and_mask(
                            extra_img,
                            extra_mask,
                            target_size=(input_height, input_width),
                        )
                        yield _preprocess_image_and_mask(
                            extra_img,
                            extra_mask,
                            target_size=(input_height, input_width),
                            transform=transform,
                        )

    return tf.data.Dataset.from_generator(
        my_generator,
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
            folder_path=folder_path,
            augumentation_operations=operations,
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
