"""Convience script to launch a run to kubeflow. Handles the experiment, pipeline, pipeline version, and run creation as well as building an image if necessary."""
from absl import flags
from absl import logging
from typing import Optional, Text

import os
import sys
from mlp.utils.dirs import pipeline_var_names
import importlib.util
import datetime
from mlp.utils.config import VarConfig
from mlp.utils.docker import (
    build_image,
    push_image,
    local_image_exists,
    remote_image_exists,
)
from mlp.utils.kubeflow import get_pipeline_version_id, run_pipeline_file
import kfp


def check_environment_set():
    environs = [
        "GOOGLE_APPLICATION_CREDENTIALS",
        "PROJECT",
        "CLIENT_ID",
        "CLIENT_SECRET",
        "OTHER_CLIENT_ID",
        "OTHER_CLIENT_SECRET",
        "ENDPOINT",
        "NAMESPACE",
        "MLP_VERSION",
        "PROJECT_VERSION",
    ]
    for environ in environs:
        if os.environ.get(environ) is None or os.environ.get(environ).strip() == "":
            sys.exit(
                "{environ} not set. Must set to appropriate value before runnign this script.".format(
                    environ=environ
                )
            )


def create_update_run(pipeline_path: Text, experiment: Optional[Text] = "dev"):
    check_environment_set()

    # Pull in the pipeline file as a module to get various variables.
    # pipeline_path = FLAGS.pipeline_path
    pipeline_mod = pipeline_path.split("/")[-1].rstrip(".py")
    spec = importlib.util.spec_from_file_location(pipeline_mod, pipeline_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Create a unique identifier for the run.
    run_str = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")

    # Get the standard variable names
    var_names = pipeline_var_names(
        mod._RUN_DIR,
        run_str,
        mod._MLP_PROJECT,
        mod._MLP_SUBPROJECT,
        mod._RUNNER,
        mod._PIPELINE_TYPE,
        experiment,
    )
    # Run the pipeline file to get the pipeline package to upload to kubeflow
    run_pipeline_file(pipeline_path, run_str, experiment)

    # Load the config file from gcp (generated from run_pipeline_file)
    vc = VarConfig()
    vc._load_from_json(var_names["vc_config_path"])

    pipeline_package_path = vc.pipeline_name + ".tar.gz"
    image_name = "gcr.io/{gcp_project}/{mlp_project}".format(
        gcp_project=os.environ.get("PROJECT"),
        mlp_project=vc.mlp_project,
    )
    pipeline_name = var_names["pipeline_name"]

    # Create and push the image if it doesn't already exist on gcp
    if remote_image_exists(image_name, vc.version):
        logging.warning(
            "Image {}:{} already exists remotely so it won't be built. If you have changes to local project they may not be in the remote image. Increment the version number in version.py to build a new image with the current changes.".format(
                image_name, vc.version
            )
        )
    else:
        if not local_image_exists(image_name, vc.version):
            logging.info(
                "{image_name}:{image_tag} not found, building...".format(
                    image_name=image_name, image_tag=vc.version
                )
            )
            exit_code = build_image(
                image_name, vc.version, mlp_version=os.environ.get("MLP_VERSION")
            )
            if exit_code != 0:
                sys.exit(
                    "Failed to build image {image_name}:{image_tag}. Try building yourself with 'docker build . -t {image_name}:{image_tag}' to see what's wrong".format(
                        image_name=image_name, image_tag=vc.version
                    )
                )

        exit_code = push_image(image_name, vc.version)
        if exit_code != 0:
            sys.exit(
                "Failed to push image {image_name}:{image_tag}. Try pushing yourself with 'docker push {image_name}:{image_tag} to see what's wrong.".format(
                    image_name=image_name, image_tag=vc.version
                )
            )

    # Define the kubeflow client
    client = kfp.Client(
        host=os.environ.get("ENDPOINT"),
        client_id=os.environ.get("CLIENT_ID"),
        namespace=os.environ.get("NAMESPACE"),
        other_client_id=os.environ.get("OTHER_CLIENT_ID"),
        other_client_secret=os.environ.get("OTHER_CLIENT_SECRET"),
    )

    # Create experiment if it doesn't exist
    experiment_objs = client.list_experiments(page_size=100000).experiments
    experiment_objs = experiment_objs if experiment_objs is not None else []
    experiments = [d.name for d in experiment_objs]
    if experiment not in experiments:
        client.create_experiment(experiment)
    experiment_id = client.get_experiment(experiment_name=experiment).id

    # Upload pipeline if it doesn't exist
    pipeline_objs = client.list_pipelines(page_size=100000).pipelines
    pipeline_objs = pipeline_objs if pipeline_objs is not None else []
    pipelines = [d.name for d in pipeline_objs]
    if pipeline_name not in pipelines:
        client.upload_pipeline(pipeline_package_path, pipeline_name=pipeline_name)
    pipeline_id = client.get_pipeline_id(pipeline_name)

    # Upload pipeline version if it doesn't exist
    pipeline_version = vc.version + "-" + vc.hash
    pipeline_versions = [
        d.name
        for d in client.list_pipeline_versions(pipeline_id, page_size=100000).versions
    ]
    if pipeline_version not in pipeline_versions:
        client.upload_pipeline_version(
            pipeline_package_path,
            pipeline_version_name=pipeline_version,
            pipeline_id=pipeline_id,
        )
    pipeline_version_id = get_pipeline_version_id(
        client, pipeline_name, pipeline_version
    )

    # Launch run
    job_name = pipeline_name + "-" + pipeline_version + "-" + run_str
    client.run_pipeline(
        experiment_id=experiment_id,
        job_name=job_name,
        version_id=pipeline_version_id,
    )
