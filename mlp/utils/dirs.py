import os
import datetime


def pipeline_dirs(run_dir, run_str, mlp_project, mlp_subproject, pipeline_name):
  proj_root = os.path.join(run_dir, 'tfx', pipeline_name)

  if run_str is not None:
    run_root = os.path.join(proj_root, run_str)
  else:
    run_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    run_root = os.path.join(proj_root, run_str)

  model_uri = os.path.join(run_dir, 'serving', mlp_project, mlp_subproject, 'model')
  schema_uri = os.path.join(run_dir, 'serving', mlp_project, mlp_subproject, 'schema')
  transform_graph_uri = os.path.join(run_dir, 'serving', mlp_project, mlp_subproject, 'transform_graph')
  pipeline_root = os.path.join(run_root, 'data')

  return pipeline_root, model_uri, schema_uri, transform_graph_uri
