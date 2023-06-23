"""Simple helper functions for dealing with kubeflow and kfp."""
from typing import Text, Optional
from absl import logging
from subprocess import Popen, PIPE

import sys
import os
import kfp
from mlp.utils.dirs import pipeline_var_names


def get_pipeline_version_id(
    client: kfp.Client,
    pipeline_name: Text,
    pipeline_version_name: Text,
    page_size=100000,
) -> Text:
    """Get the pipeline version id from the pipeline name and pipeline version name."""
    pipeline_version_id = None

    pipeline_id = client.get_pipeline_id(pipeline_name)
    for d in client.list_pipeline_versions(pipeline_id, page_size=page_size).versions:
        if d.name == pipeline_version_name:
            pipeline_version_id = d.id

    if pipeline_version_id is None:
        sys.exit(
            "pipeline with pipeline_name = {}, pipeline_version_name = {} not found on kfp".format(
                pipeline_name, pipeline_version_name
            )
        )

    return pipeline_version_id


def run_pipeline_file(
  pipeline_path: Text,
  project_dir: Text,
  run_str: Text,
  version: Text,
  experiment: Optional[Text] = None):
    """Create the pipeline tar file to upload to kubeflow."""
    if os.path.isdir(pipeline_path):
        sys.exit("Provide pipeline file path.")

    # Run dsl with mock environment to store pipeline args in temp_file.
    if experiment:
        command = [sys.executable, pipeline_path, run_str, experiment, version, project_dir]
    else:
        command = [sys.executable, pipeline_path, run_str, version, project_dir]

    p = Popen(command, stdout=PIPE, stderr=PIPE)

    stdout, stderr = p.communicate()

    logging.info("run_pipeline_file stdout:")
    logging.info(stderr)
    logging.error("run_pipeline_file stderr:")
    logging.error(stdout)

    if p.returncode != 0:
        sys.exit('Error while running "{}" '.format(" ".join(command)))


def get_beam_and_kubeflow_config(var_config):
    vars = {}
    vars.update(
        pipeline_var_names(
            var_config.run_dir,
            var_config.run_str,
            var_config.mlp_project,
            var_config.mlp_subproject,
            var_config.runner,
            var_config.pipeline_type,
            var_config.experiment,
        )
    )

    schema_path = [vars["run_root"]] + var_config.schema_params
    vars["schema_uri"] = os.path.join(*schema_path)
    vars["image_name"] = "gcr.io/{gcp_project}/{mlp_project}:{version}".format(
        gcp_project=var_config.gcp_project,
        mlp_project=var_config.mlp_project,
        version=var_config.version,
    )

    # If running with dataflow
    vars["beam_pipeline_args"] = var_config.beam_pipeline_args + [
        "--project=" + var_config.gcp_project,
        "--temp_location=" + os.path.join(vars["run_root"], "tmp"),
        "--region=" + var_config.gcp_region,
        "--sdk_container_image=" + vars["image_name"],
    ]
    vars["hash"] = var_config.get_hash()

    return vars
