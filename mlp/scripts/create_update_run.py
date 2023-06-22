"""Convience script to launch a run to kubeflow.
Handles the experiment, pipeline, pipeline version,
and run creation as well as building an image if necessary."""
from absl import logging
from typing import Optional, Text

import os
import sys
import datetime
from mlp.utils.docker import (
    build_image,
    push_image,
    local_image_exists,
    remote_image_exists,
)
from mlp.utils.kubeflow import get_pipeline_version_id, run_pipeline_file
import kfp
from git import Repo
import re


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
        "MLP_VERSION"
    ]
    for environ in environs:
        if os.environ.get(environ) is None or os.environ.get(environ).strip() == "":
            sys.exit(
                "{environ} not set. Must set to appropriate value before runnign this script."
                .format(environ=environ)
            )


def create_update_run(
        pipeline_path: Text, project_dir: Text, pipeline_name: Text,
        pipeline_docker_path: Text, experiment: Optional[Text] = "dev",
        update: Optional[bool] = False, upgrade: Optional[bool] = False):
    check_environment_set()
    # Create a unique identifier for the run.
    run_str = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")

    repo = Repo(search_parent_directories=True)
    pattern = r'.*\.py$'  # regex pattern to match python files
    exclude_pattern = r'.*(setup|__init__)\.py$'
    changed_files = [item.a_path for item in repo.index.diff(None) + repo.index.diff('HEAD')]
    changed_flag = False
    for f in changed_files:
        if re.match(pattern, f) and not re.match(exclude_pattern, f):
            changed_flag = True
            break

    # Define the kubeflow client
    client = kfp.Client(
        host=os.environ.get("ENDPOINT"),
        client_id=os.environ.get("CLIENT_ID"),
        namespace=os.environ.get("NAMESPACE"),
        other_client_id=os.environ.get("OTHER_CLIENT_ID"),
        other_client_secret=os.environ.get("OTHER_CLIENT_SECRET"),
    )
    pipeline_id = client.get_pipeline_id(pipeline_name)
    pipeline_versions = []
    pipelines = client.list_pipeline_versions(pipeline_id, sort_by="created_at desc").versions
    if pipeline_id and pipelines:
        # Upload pipeline version if it doesn't exist
        pipeline_versions = [d.name for d in pipelines]
        latest_version = pipeline_versions[0]
        if changed_flag:
            parts = latest_version.split('-')[0].split(".")
            if upgrade:
                parts[-3] = str(int(parts[-3]) + 1)
                auto_inc_version = parts[-3] + '.0.0'
            elif update:
                parts[-2] = str(int(parts[-2]) + 1)
                auto_inc_version = parts[-3] + '.' + parts[-2] + '.0'
            else:
                parts[-1] = str(int(parts[-1]) + 1)
                auto_inc_version = ".".join(parts)
        else:
            auto_inc_version = latest_version
    else:
        auto_inc_version = "0.0.1"

    # Run the pipeline file to get the pipeline package to upload to kubeflow
    run_pipeline_file(pipeline_path, project_dir, run_str, auto_inc_version, experiment)
    context_path = os.path.dirname(pipeline_docker_path)
    pipeline_package_path = context_path + '/' + pipeline_name + ".tar.gz"
    image_name = "gcr.io/{gcp_project}/{pipeline_name}".format(
        gcp_project=os.environ.get("PROJECT"),
        pipeline_name=pipeline_name,
    )

    # Create and push the image if it doesn't already exist on gcp
    if remote_image_exists(image_name, auto_inc_version):
        logging.warning(
            "Image {}:{} already exists remotely it won't be built"
            .format(image_name, auto_inc_version) +
            ". If you have changes to local project they may not be in the remote image" +
            ". Increment the version number in version.py to build a new image" +
            " with the current changes."
        )
    else:
        if not local_image_exists(image_name, auto_inc_version):
            logging.info(
                "{image_name}:{image_tag} not found, building...".format(
                    image_name=image_name, image_tag=auto_inc_version
                )
            )
            exit_code = build_image(
                image_name,
                auto_inc_version,
                mlp_version=os.environ.get("MLP_VERSION"),
                dir=context_path,
                docker_file=pipeline_docker_path
            )
            if exit_code != 0:
                sys.exit(
                    "Failed to build image {image_name}:{image_tag}. "
                    .format(image_name=image_name, image_tag=auto_inc_version) +
                    "Try building yourself with 'docker build . -t {image_name}:{image_tag}'"
                    .format(image_name=image_name, image_tag=auto_inc_version) +
                    " to see what's wrong"
                )

        exit_code = push_image(image_name, auto_inc_version)
        if exit_code != 0:
            sys.exit(
                "Failed to push image {image_name}:{image_tag}. ".format(
                    image_name=image_name, image_tag=auto_inc_version
                ) +
                "Try pushing yourself with 'docker push {image_name}:{image_tag}".format(
                    image_name=image_name, image_tag=auto_inc_version
                )
                + " to see what's wrong."
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

    if auto_inc_version not in pipeline_versions:
        client.upload_pipeline_version(
            pipeline_package_path,
            pipeline_version_name=auto_inc_version,
            pipeline_id=pipeline_id,
        )
    pipeline_version_id = get_pipeline_version_id(
        client, pipeline_name, auto_inc_version
    )

    # Launch run
    job_name = pipeline_name + '-' + auto_inc_version
    run = client.run_pipeline(
        experiment_id=experiment_id,
        job_name=job_name,
        version_id=pipeline_version_id
    )
    if changed_flag:
        repo.git.commit(a=True, m='pipeline changed, run name: ' + run.name)
