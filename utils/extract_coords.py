from PIL import Image
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN


def extract_coords(image: Path | Image.Image):
    bounding_boxes = []
    if isinstance(image, Path):
        image = Image.open(image)
    pixels = np.array(image)

    # Define red color threshold
    red_threshold = (
        (pixels[:, :, 0] > 150) & (pixels[:, :, 1] < 100) & (pixels[:, :, 2] < 100)
    )

    # Get coordinates of red pixels
    red_coords = np.column_stack(np.where(red_threshold))
    if red_coords.size > 0:
        cluster = DBSCAN(eps=10, min_samples=5).fit(red_coords)
        unique_labels = set(cluster.labels_)
        for label in unique_labels:
            if label != -1:
                cluster_points = red_coords[cluster.labels_ == label]
                min_y, min_x = cluster_points.min(axis=0)
                max_y, max_x = cluster_points.max(axis=0)
                bounding_boxes.append((int(min_x), int(min_y), int(max_x), int(max_y)))
    return bounding_boxes


from PIL import ImageDraw


def check_result_correct(img_path: Path, bounding_boxes):
    img = Image.open(img_path)
    img_draw = ImageDraw.Draw(img)
    for min_x, min_y, max_x, max_y in bounding_boxes:
        img_draw.rectangle([min_x, min_y, max_x, max_y], outline="yellow", width=2)
    return img


def resize_image(img, new_width=640):
    # Open the image

    # Calculate new height maintaining aspect ratio
    width_percent = new_width / float(img.width)
    new_height = int(float(img.height) * width_percent)

    # Resize the image
    resized_img = img.resize((new_width, new_height))

    return resized_img


def rotate_images(origin_file, marked_file):
    origin_img = Image.open(origin_file)
    marked_img = Image.open(marked_file)
    origin_width, origin_height = origin_img.width, origin_img.height
    marked_width, marked_height = marked_img.width, marked_img.height
    if marked_height > marked_width:
        marked_img = marked_img.transpose(Image.Transpose.ROTATE_90)
    marked_width, marked_height = marked_img.width, marked_img.height
    if origin_width != marked_width:
        origin_img = origin_img.transpose(Image.Transpose.ROTATE_90)
    return origin_img, marked_img


def rotate_normalized_coords(x_center, y_center, width, height, angle):
    match angle:
        case 90:
            return y_center, 1 - x_center, height, width
        case 180:
            return 1 - x_center, 1 - y_center, width, height
        case 270:
            return 1 - y_center, x_center, height, width
        case _:
            raise ValueError("Angle must be one of 90, 180, 270")


def calculate_overlap(pic: Image.Image, slice_coords, bounding_box):
    """
    Calculate the overlap ratio between a slice and a YOLO bounding box.

    Parameters:
    - pic: PIL Image object
    - slice_coords: (min_x, min_y, max_x, max_y) in absolute pixel coordinates
    - bounding_box: (x_center, y_center, width, height) in YOLO normalized format

    Returns:
    - overlap_ratio: overlap area / bounding box area
    """
    image_width, image_height = pic.size
    x_center, y_center, width, height = bounding_box

    # Convert YOLO normalized coordinates to absolute pixel coordinates
    box_width = width * image_width
    box_height = height * image_height
    box_x_center = x_center * image_width
    box_y_center = y_center * image_height

    box_min_x = box_x_center - box_width / 2
    box_max_x = box_x_center + box_width / 2
    box_min_y = box_y_center - box_height / 2
    box_max_y = box_y_center + box_height / 2

    # Slice coordinates
    slice_min_x, slice_min_y, slice_max_x, slice_max_y = slice_coords

    # Calculate intersection rectangle
    inter_min_x = max(box_min_x, slice_min_x)
    inter_max_x = min(box_max_x, slice_max_x)
    inter_min_y = max(box_min_y, slice_min_y)
    inter_max_y = min(box_max_y, slice_max_y)

    # Compute intersection area
    inter_width = max(0, inter_max_x - inter_min_x)
    inter_height = max(0, inter_max_y - inter_min_y)
    inter_area = inter_width * inter_height

    # Compute bounding box area
    box_area = box_width * box_height

    # Compute overlap ratio
    overlap_ratio = inter_area / box_area if box_area > 0 else 0

    return overlap_ratio


def img_slices_coords(pic: Image.Image, step_size=4):
    if pic.height > pic.width:
        raise ValueError("Please ensure the width >= height")
    coords = []
    step = int(pic.height / step_size)
    min_x = 0
    min_y = 0
    max_y = pic.height
    max_x = pic.height

    while max_x < pic.width:
        coords.append((min_x, min_y, max_x, max_y))
        min_x += step
        max_x += step
    return coords
