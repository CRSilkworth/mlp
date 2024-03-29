import os
import datetime
from google.cloud import storage
from typing import Tuple, Text, Optional, List
import tensorflow as tf


def pipeline_dirs(
    run_dir: Text,
    run_str: Text,
    mlp_project: Text,
    mlp_subproject: Text,
    pipeline_name: Text,
    experiment: Optional[Text] = None,
) -> Tuple[Text, Text, Text, Text]:
    """Get the standard directory names from the project names.

    Parameters
    ----------
    run_dir: Directory to store all pipeline runs.
    run_str: The unique identifier of the
    mlp_project: dskjl;
    pipeline_name: dfi

    Returns
    -------
    proj_root: The root directory of the mlp_project.
    run_root: The root directory of one run of the mlp_subproject.
    pipeline_root: The directory to write all the artifacts/metadata for the kubeflow pipeline.
    serving_uri: The directory to write all final output objects (e.g. trained model.)

    """
    if experiment is None:
        pipeline_root = os.path.join(run_dir, "tfx", pipeline_name)
        serving_root = os.path.join(run_dir, "serving")
    else:
        pipeline_root = os.path.join(run_dir, "tfx", experiment, pipeline_name)
        serving_root = os.path.join(run_dir, "serving", experiment)

    if run_str is not None:
        run_root = os.path.join(pipeline_root, run_str)
    else:
        run_str = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
        run_root = os.path.join(pipeline_root, run_str)

    data_root = os.path.join(run_root, "data")

    return pipeline_root, run_root, data_root, serving_root


def pipeline_var_names(
    run_dir: Text,
    run_str: Text,
    mlp_project: Text,
    mlp_subproject: Text,
    runner: Text,
    pipeline_type: Text,
    experiment: Optional[Text] = None,
):
    pipeline_name = "-".join([mlp_project, mlp_subproject, pipeline_type])

    date = datetime.datetime.now().strftime("%Y_%m_%d")
    kfp_pipeline_name = "-".join([pipeline_name, date])

    pipeline_mod = ".".join([mlp_subproject, "pipelines", runner, pipeline_type])

    pipeline_root, run_root, data_root, serving_root = pipeline_dirs(
        run_dir,
        run_str,
        mlp_project,
        mlp_subproject,
        pipeline_name,
        experiment=experiment,
    )

    return {
        "pipeline_root": pipeline_root,
        "run_root": run_root,
        "data_root": data_root,
        "serving_root": serving_root,
        "serving_dir": os.path.join(serving_root, mlp_project, mlp_subproject),
        "pipeline_mod": pipeline_mod,
        "pipeline_name": pipeline_name,
        "kfp_pipeline_name": kfp_pipeline_name,
        "vc_config_path": os.path.join(run_root, "config", "pipeline_vars.json"),
        "metadata_path": os.path.join(data_root, "metadata", "metadata.db"),
    }


def lstrip_dirs(full_path, rm_path):
    full_dirs = [d for d in full_path.split("/") if d != ""]
    rm_dirs = [d for d in rm_path.split("/") if d != ""]

    while rm_dirs:
        rm_dir = rm_dirs.pop(0)
        full_dir = full_dirs.pop(0)

        if rm_dir != full_dir:
            raise ValueError("{} not a prefix of {}.".format(rm_path, full_path))

    if rm_dirs:
        raise ValueError("{} not a prefix of {}.".format(rm_path, full_path))

    return "/".join(full_dirs)


def copy_dir(src_uri: Text, dst_uri: Text, ignore_subdirs: Optional[List[Text]] = None):
    """Copy directory to/from gcp and local."""

    if ignore_subdirs is None:
        ignore_subdirs = []
    else:
        ignore_subdirs = [os.path.join(src_uri, d) for d in ignore_subdirs]

    tf.io.gfile.makedirs(dst_uri)

    for src_dir_name, sub_dirs, file_names in tf.io.gfile.walk(src_uri):
        skip_dir = False
        for ignore_subdir in ignore_subdirs:
            if src_dir_name.startswith(ignore_subdir):
                skip_dir = True
                break

        if skip_dir:
            continue

        dst_dir_name = os.path.join(dst_uri, lstrip_dirs(src_dir_name, src_uri))
        tf.io.gfile.makedirs(dst_dir_name)

        for file_name in file_names:
            src_file_name = os.path.join(src_dir_name, file_name)
            dst_file_name = os.path.join(dst_dir_name, file_name)
            tf.io.gfile.copy(src_file_name, dst_file_name)


def download_dir(
    src_uri: Text, dst_uri: Text, ignore_subdirs: Optional[List[Text]] = None
):
    """Download directory from gcp bucket."""
    if ignore_subdirs is None:
        ignore_subdirs = []
    else:
        ignore_subdirs = [os.path.join(src_uri, d) for d in ignore_subdirs]

    bucket_name = src_uri.replace("gs://", "").split("/")[0]
    prefix = "/".join(src_uri.replace("gs://", "").split("/")[1:])

    if not bucket_name or not src_uri.startswith("gs://"):
        raise ValueError(
            "src_uri must start with gs://<bucket_name>. Got {}".format(src_uri)
        )

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        src_file_name = blob.name.split("/")[-1]
        if not src_file_name:
            continue

        src_dir_name = os.path.join(
            "gs://", bucket_name, "/".join(blob.name.split("/")[:-1])
        )
        skip_dir = False

        for ignore_subdir in ignore_subdirs:
            if src_dir_name.startswith(ignore_subdir):
                skip_dir = True
                break

        if skip_dir:
            continue

        dst_dir_name = os.path.join(dst_uri, lstrip_dirs(src_dir_name, src_uri))
        tf.io.gfile.makedirs(dst_dir_name)

        print(dst_dir_name, src_file_name)
        dst_file_name = os.path.join(dst_dir_name, src_file_name)
        blob.download_to_filename(dst_file_name)
