from pathlib import Path
import numpy as np
from pydicom import dcmread
from PIL import Image
import keras
import pandas as pd


def imgs_generator(src_folder: Path, batch_size=64):
    batch = []
    for file in src_folder.rglob("*"):
        if file.is_file() and file.suffix.lower() == ".dcm":
            batch.append(file)
            if len(batch) == batch_size:
                yield batch
                batch = []
    yield batch


def label_agg_func(labels):
    valid_tags = set(("合格焊缝区域", "非焊缝区域"))
    labels = set(labels)
    all_valid = all([label in valid_tags for label in labels])
    if all_valid:
        return "合格"
    else:
        return ";".join([label for label in labels if label not in valid_tags])


def process_dicom_images(src_folder: Path, model_path: Path, labels_config, callback):
    model = eval_model(model_path)
    final_df = pd.DataFrame()
    for index, batch_dcm in enumerate(imgs_generator(src_folder)):
        file_labels = []
        result = load_dcm_into_slices(batch_dcm)
        imgs = [img for (_, img) in result]
        files = [file for (file, _) in result]
        imgs = np.array(imgs)
        labels_and_confidence = model_predict(model, imgs, labels_config)
        for file, (label, confidence) in zip(files, labels_and_confidence):
            file_labels.append((file, label, confidence))
        df = callback(index, file_labels, src_folder)
        final_df = pd.concat([final_df, df])

    final_df = (
        final_df[["file_name", "label"]]
        .groupby("file_name")
        .agg({"label": label_agg_func})
    ).reset_index()

    rows_to_check_index = ~final_df.apply(
        lambda row: any(label in row["file_name"] for label in row["label"].split(";")),
        axis=1,
    )

    return final_df, rows_to_check_index


def img_process(img_array):
    imgs = []
    height, width = img_array.shape
    if height > width:
        img_array = img_array.transpose()
    height, width = img_array.shape
    step = int(height / 4)
    start_width = 0
    while start_width + height < width:
        img = img_array[:, start_width : (start_width + height)]
        img = Image.fromarray(img).resize((512, 512))
        img = np.asarray(img)
        img = img.reshape(*img.shape, 1)
        imgs.append(img)
        start_width += step
    imgs = np.array(imgs)
    return imgs


def load_dcm_into_slices(dcm_paths: Path) -> np.array:
    result = []
    for dcm_path in dcm_paths:
        img_array = dcmread(dcm_path).pixel_array
        img_array = (img_array / img_array.max()) * 255.0
        img_array = img_array.astype("uint8")
        for img in img_process(img_array):
            result.append((dcm_path, img))
    return result


def eval_model(model_path: Path):
    model = keras.models.load_model(model_path, compile=False)
    return model


def model_predict(model: keras.Model, imgs: np.array, labels_config):
    predictions = model.predict(imgs)
    label_indices = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    return [
        (labels_config[label_index], confidence)
        for (label_index, confidence) in zip(label_indices, confidences)
    ]


def aggregate_batches(src_folder: Path, batch_prefix: str, result_filename=None):
    df = pd.DataFrame()
    for file in src_folder.rglob(f"{batch_prefix}*"):
        df = pd.concat([df, pd.read_excel(file)])
        file.unlink()
    if result_filename:
        df.to_excel(src_folder / result_filename, index=False)
    return df


from .models import RootFolder, SubFolder, Sess, engine, Base, WorkStatus
