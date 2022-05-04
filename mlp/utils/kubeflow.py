from typing import Any, Optional, Text, Dict
from absl import logging
from subprocess import Popen, PIPE

import sys
import os


def get_pipeline_version_id(client, pipeline_name, pipeline_version_name):
  """Get the pipeline version id from the pipeline name and pipeline version name."""

  pipeline_version_id = None

  pipeline_id = client.get_pipeline_id(pipeline_name)
  for d in client.list_pipeline_versions(pipeline_id).versions:
    if d.name == pipeline_version_name:
      pipeline_version_id = d.id

  if pipeline_version_id is None:
    sys.exit('pipeline with pipeline_name = {}, pipeline_version_name = {} not found on kfp'.format(pipeline_name, pipeline_version_name))

  return pipeline_version_id


def run_pipeline_file(pipeline_path, run_str, experiment=None):
  """Create the pipeline tar file to upload to kubeflow."""
  if os.path.isdir(pipeline_path):
    sys.exit('Provide pipeline file path.')

  # Run dsl with mock environment to store pipeline args in temp_file.
  if experiment:
    command = [sys.executable, pipeline_path, run_str, experiment]
  else:
    command = [sys.executable, pipeline_path, run_str]

  p = Popen(command, stdout=PIPE, stderr=PIPE)

  stdout, stderr = p.communicate()

  logging.info('run_pipeline_file stdout:')
  logging.info(stderr)
  logging.error('run_pipeline_file stderr:')
  logging.error(stdout)

  if p.returncode != 0:
    sys.exit('Error while running "{}" '.format(' '.join(command)))
