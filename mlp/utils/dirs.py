import os
import datetime


def pipeline_dirs(run_dir, run_str, mlp_project, mlp_subproject, pipeline_name):
  proj_root = os.path.join(run_dir, 'tfx', pipeline_name)

  if run_str is not None:
    run_root = os.path.join(proj_root, run_str)
  else:
    run_str = datetime.datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S')
    run_root = os.path.join(proj_root, run_str)

  serving_uri = os.path.join(run_root, 'serving')
  pipeline_root = os.path.join(run_root, 'data')

  return proj_root, run_root, pipeline_root, serving_uri
