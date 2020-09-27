import os
import datetime
from typing import Tuple, Text


def pipeline_dirs(
  run_dir: Text,
  run_str: Text,
  mlp_project: Text,
  mlp_subproject: Text,
  pipeline_name: Text) -> Tuple[Text, Text, Text, Text]:
  proj_root = os.path.join(run_dir, 'tfx', pipeline_name)
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
  if run_str is not None:
    run_root = os.path.join(proj_root, run_str)
  else:
    run_str = datetime.datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S')
    run_root = os.path.join(proj_root, run_str)

  serving_uri = os.path.join(run_root, 'serving')
  pipeline_root = os.path.join(run_root, 'data')

  return proj_root, run_root, pipeline_root, serving_uri
