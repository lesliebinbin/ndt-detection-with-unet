from skimage import feature, exposure, transform
from pathlib import Path
from PIL import Image
import numpy as np
from pydicom import dcmread
import tensorflow as tf


def remove_white_theshold(
    image: Path | Image.Image | np.ndarray,
    padding=10,
    low_threshold_ratio=0.1,
    high_threshold_ratio=0.2,
):
    if isinstance(image, Path):
        if image.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            img = np.array(Image.open(image))
        elif image.suffix.lower() == ".dcm":
            img = dcmread(image).pixel_array
        else:
            raise ValueError("Unsupported file format")
    elif isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image
    img_max = img.max()
    low_threshold = img_max * low_threshold_ratio
    high_threshold = img_max * high_threshold_ratio
    edges = feature.canny(
        img, sigma=2.0, low_threshold=low_threshold, high_threshold=high_threshold
    )
    rows, cols = np.where(edges)
    if len(rows) == 0 or len(cols) == 0:
        return img, False
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    min_row = max(0, min_row - padding)
    max_row = min(img.shape[0], max_row + padding)
    min_col = max(0, min_col - padding)
    max_col = min(img.shape[1], max_col + padding)

    return img[min_row:max_row, min_col:max_col], True


import numpy as np


def create_square_image(image: np.ndarray):
    height, width = image.shape
    square_length = height + width
    square_image = np.zeros((square_length, square_length), dtype=image.dtype)

    # 放置原始图像在左上角
    square_image[0:height, 0:width] = image

    # 顺时针旋转90度并放置在左下角
    rot_90_image = np.rot90(image, k=-1)
    square_image[height : height + rot_90_image.shape[0], 0 : rot_90_image.shape[1]] = (
        rot_90_image
    )

    # 旋转180度并放置在右下角
    rot_180_image = np.rot90(image, k=2)
    square_image[
        height : height + rot_180_image.shape[0], width : width + rot_180_image.shape[1]
    ] = rot_180_image

    # 逆时针旋转90度(相当于顺时针270度)并放置在右上角
    rot_270_image = np.rot90(image, k=1)
    square_image[0 : rot_270_image.shape[0], width : width + rot_270_image.shape[1]] = (
        rot_270_image
    )

    return square_image


def extract_windowed_region(image: np.ndarray, window_center, window_width):
    window_min = window_center - window_width // 2
    window_max = window_center + window_width // 2
    mask = (image >= window_min) & (image <= window_max)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return image, False
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    roi = image[rmin : rmax + 1, cmin : cmax + 1]

    windowed = exposure.rescale_intensity(
        roi, in_range=(window_min, window_max), out_range=(0, 255)
    ).astype(np.uint8)
    windowed = transform.resize(windowed, (image.shape[0], image.shape[1]))
    return windowed, True


def create_artificial_images(dcm_path: Path):
    origin_image = remove_white_theshold(dcm_path)[0]
    windowed_center = origin_image.mean()
    windowed_std = origin_image.std()
    scale_1_image = extract_windowed_region(
        origin_image, window_center=windowed_center, window_width=2 * windowed_std
    )[0]
    scale_2_image = extract_windowed_region(
        origin_image, window_center=windowed_center, window_width=4 * windowed_std
    )[0]
    artificial_image = np.stack([origin_image, scale_1_image, scale_2_image]).transpose(
        (1, 2, 0)
    )
    artificial_image = (
        artificial_image / artificial_image.max(axis=(0, 1)) * 255
    ).astype(np.uint8)
    return artificial_image


def create_custom_three_channel_img(dcm_path: Path):
    img = remove_white_theshold(dcm_path)[0]
    split = img.max() // 3
    red_channel_mask = np.where(img <= split, 1, 0)
    red_channel = ((img * red_channel_mask / split) * 255).astype(np.uint8)
    green_channel_mask = np.where((split < img) & (img <= 2 * split), 1, 0)
    green_channel = (img * green_channel_mask - split) / split * 255
    green_channel = np.where(green_channel >= 0, green_channel, 0).astype(np.uint8)
    blue_channel = np.where(img > 2 * split, img, 0)
    blue_channel = (img * blue_channel - 2 * split) / split * 255
    blue_channel = np.where(blue_channel >= 0, blue_channel, 0).astype(np.uint8)
    return np.stack([green_channel, red_channel, blue_channel]).transpose((1, 2, 0))


def calculate_union(bounding_boxes):
    x_min, y_min, x_max, y_max = [], [], [], []
    for x_center, y_center, width, height in bounding_boxes:
        x_min.append(x_center - width / 2)
        y_min.append(y_center - height / 2)
        x_max.append(x_center + width / 2)
        y_max.append(y_center + height / 2)
    x_min = min(x_min)
    y_min = min(y_min)
    x_max = max(x_max)
    y_max = max(y_max)
    return (x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min


def check_within(bbox1, bbox2):
    def to_corners(x, y, w, h):
        x_min = x - w / 2
        x_max = x + w / 2
        y_min = y - h / 2
        y_max = y + h / 2
        return x_min, y_min, x_max, y_max

    x1_min, y1_min, x1_max, y1_max = to_corners(*bbox1)
    x2_min, y2_min, x2_max, y2_max = to_corners(*bbox2)

    return (
        x2_min >= x1_min and x2_max <= x1_max and y2_min >= y1_min and y2_max <= y1_max
    )


def calculate_overlap(bbox1, bbox2):
    def to_corners(x, y, w, h):
        x_min = x - w / 2
        x_max = x + w / 2
        y_min = y - h / 2
        y_max = y + h / 2
        return x_min, y_min, x_max, y_max

    x1_min, y1_min, x1_max, y1_max = to_corners(*bbox1)
    x2_min, y2_min, x2_max, y2_max = to_corners(*bbox2)
    new_x_min = max(x1_min, x2_min)
    new_x_max = min(x1_max, x2_max)
    new_y_min = max(y1_min, y2_min)
    new_y_max = min(y1_max, y2_max)
    if new_x_max <= new_x_min or new_y_max <= new_y_min:
        return (0., 0., 0., 0.)
    return (
        (new_x_min + new_x_max) / 2,
        (new_y_min + new_y_max) / 2,
        new_x_max - new_x_min,
        new_y_max - new_y_min,
    )


def crop_image_and_calculate_new_bbox(img: Image.Image, bbox1, bbox2):
    W, H = img.width, img.height

    # Convert bbox1 to pixel coordinates
    x1, y1, w1, h1 = bbox1
    left = int((x1 - w1 / 2) * W)
    right = int((x1 + w1 / 2) * W)
    top = int((y1 - h1 / 2) * H)
    bottom = int((y1 + h1 / 2) * H)

    cropped_img = img.crop((left, top, right, bottom))
    cropped_W, cropped_H = cropped_img.size

    # Convert bbox2 to pixel coordinates
    x2, y2, w2, h2 = bbox2
    x2_pixel = x2 * W
    y2_pixel = y2 * H
    w2_pixel = w2 * W
    h2_pixel = h2 * H

    # Adjust bbox2 relative to cropped image
    new_x = (x2_pixel - left) / cropped_W
    new_y = (y2_pixel - top) / cropped_H
    new_w = w2_pixel / cropped_W
    new_h = h2_pixel / cropped_H

    return cropped_img, (new_x, new_y, new_w, new_h)

