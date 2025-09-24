#!/usr/bin/env python
import os
import traceback
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
from apscheduler.executors.pool import ThreadPoolExecutor

os.environ["KERAS_BACKEND"] = "jax"
from workers import (
    process_dicom_images,
    aggregate_batches,
    Sess,
    engine,
    RootFolder,
    SubFolder,
    WorkStatus,
)
import yaml
from pathlib import Path
import pandas as pd
import json


def callback_func(batch_index, file_labels, src_folder):
    columns = ["file_name", "label", "confidence"]
    rows = [
        [str(file.relative_to(src_folder)), label, confidence]
        for file, label, confidence in file_labels
    ]
    return pd.DataFrame(data=rows, columns=columns)


def generate_remote_url(df, remote_template, remote_prefix, src_folder):
    def remote_url(filepath):
        filepath = (src_folder / filepath).relative_to(src_folder.parent)
        parent_folder = filepath.parent
        result = remote_template.format(
            prefix=remote_prefix,
            filepath=filepath,
            parent_folder=parent_folder,
        )
        return result

    df["remote_url"] = df["file_name"].apply(remote_url)
    return df


def should_process(local_root_folder, src_folder):
    root_path = str(local_root_folder)
    src_folder_path = str(src_folder)
    with Sess() as session:
        root_folder = session.query(RootFolder).filter_by(path=root_path).first()
        if root_folder is None:
            root_folder = RootFolder(path=root_path)
            session.add(root_folder)
            session.commit()
            sub_folder = SubFolder(
                path=src_folder_path,
                root_folder_id=root_folder.id,
                status=WorkStatus.INIT,
            )
            session.add(sub_folder)
            session.commit()
            session.refresh(sub_folder)
            return sub_folder
        sub_folder = (
            session.query(SubFolder)
            .filter_by(root_folder_id=root_folder.id, path=src_folder_path)
            .first()
        )
        if sub_folder is None:
            sub_folder = SubFolder(
                path=src_folder_path,
                root_folder_id=root_folder.id,
                status=WorkStatus.INIT,
            )
            session.add(sub_folder)
            session.commit()
            session.refresh(sub_folder)
            return sub_folder
        if sub_folder.status == WorkStatus.FAILED:
            return sub_folder
        return None


def save_result_to_db(result: pd.DataFrame, filtered_rows, sub_folder):
    result_to_save = result.copy()
    result_to_save["sub_folder_id"] = sub_folder.id
    result_to_save["is_filtered"] = filtered_rows
    result_to_save.to_sql(
        name="classification_results",
        con=engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=100,
    )


def process_folder(
    local_root_folder,
    remote_root_folder,
    remote_template,
    model_path,
    labels_config,
    callback,
):
    for src_folder in Path(local_root_folder).iterdir():
        if src_folder.is_dir() and (
            sub_folder := should_process(local_root_folder, src_folder)
        ):
            try:
                sub_folder = SubFolder.update_status(
                    sub_folder, Sess, WorkStatus.ONGOING
                )
                print(f"Processing: {src_folder.absolute()}")
                result, filtered_rows = process_dicom_images(
                    src_folder=src_folder,
                    model_path=model_path,
                    labels_config=labels_config,
                    callback=callback,
                )
                result = generate_remote_url(
                    result,
                    remote_template=remote_template,
                    remote_prefix=remote_root_folder,
                    src_folder=src_folder,
                )
                file = src_folder / "analysis_result.xlsx"
                with pd.ExcelWriter(file, engine="openpyxl") as writer:
                    result.to_excel(writer, index=False, sheet_name="full_result")
                    result[filtered_rows].to_excel(
                        writer, index=False, sheet_name="filtered_result"
                    )
                save_result_to_db(result, filtered_rows, sub_folder)
                sub_folder = SubFolder.update_status(sub_folder, Sess, WorkStatus.DONE)
                print(f"Done: {src_folder.absolute()}")
            except Exception as e:
                print(traceback.format_exception(e))
                sub_folder = SubFolder.update_status(
                    sub_folder, Sess, WorkStatus.FAILED
                )


def execute():
    model_path = "models/resnet18_best.keras"
    with open("workers/parent_folders.yml", "r") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    with open("labels_config.json", "r") as f:
        labels_config = json.load(f)["zh"]

    labels_config = {int(k): v for k, v in labels_config.items()}

    remote_template = config["remote_template"]

    for root_folder in config["root_folders"]:
        local_root_folder = root_folder["local"]
        remote_root_folder = root_folder["remote"]

    process_folder(
        local_root_folder=local_root_folder,
        remote_root_folder=remote_root_folder,
        remote_template=remote_template,
        callback=callback_func,
        model_path=model_path,
        labels_config=labels_config,
    )


def main():
    executors = {"default": ThreadPoolExecutor(4)}

    # Schedule jobs every 30 minutes
    scheduler = BlockingScheduler(executors=executors)
    scheduler.add_job(
        execute,
        "interval",
        minutes=30,
        next_run_time=datetime.now(),
    )
    scheduler.start()


if __name__ == "__main__":
    main()