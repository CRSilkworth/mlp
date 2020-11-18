"""Run a beam pipeline that starts from pulling examples from bigquery."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from __example_subproject__.pipelines.defs.trainer_to_pusher import create_pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

from __example_subproject__ import train
import __example_subproject__.pipelines.beam.bigquery_to_pusher as full

from mlp.utils.dir import pipeline_dirs
from mlp.utils.dirs import pipeline_var_names
from mlp.utils.sql import query_with_kwargs
from mlp.utils.config import VarConfig
from mlp.utils.resolvers import latest_run_root
from mlp.utils.resolvers import latest_artifact_path
from mlp.utils.config import VarConfig
from mlp.utils.dirs import pipeline_var_names

_PIPELINE_TYPE = 'trainer_to_pusher'

trainer_fn = full.trainer_fn

if __name__ == "__main__":
  prev_run_root = '/root/runs/tfx/intent_classifier-basic-bigquery_to_pusher/2020-11-16-01-29-07'
  vc = VarConfig(os.path.join(prev_run_root, 'config', 'pipeline_vars.json'))
  vc.prev_run_root = prev_run_root
  vc.pipeline_type = _PIPELINE_TYPE
  vc.run_str = None

  vc.num_train_steps = 750
  vc.num_eval_steps = 5
  vc.warmup_prop = 0.1
  vc.cooldown_prop = 0.1
  vc.save_summary_steps = 1
  vc.save_checkpoints_secs = 14400
  vc.learning_rate = 2e-5
  vc.warm_start_from = None

  vc.model_uri = latest_artifact_path(prev_run_root, 'data/Trainer/model')
  vc.schema_uri = latest_artifact_path(prev_run_root, 'data/SchemaGen/schema')
  vc.examples_uri = latest_artifact_path(prev_run_root, 'data/Transform/transformed_examples')
  vc.transform_graph_uri = latest_artifact_path(prev_run_root, 'data/Transform/transform_graph')

  var_names = pipeline_var_names(
    vc.run_dir,
    vc.run_str,
    vc.mlp_project,
    vc.mlp_subproject,
    vc.runner,
    vc.pipeline_type
  )
  vc.add_vars(**var_names)
  vc.write(vc.vc_config_path)

  DAG = BeamDagRunner().run(
    create_pipeline(
      run_root=vc.run_root,
      pipeline_name=vc.pipeline_name,
      pipeline_mod=vc.pipeline_mod,
      schema_uri=vc.schema_uri,
      transform_graph_uri=vc.transform_graph_uri,
      examples_uri=vc.examples_uri,
      model_uri=vc.model_uri,
      beam_pipeline_args=vc.beam_pipeline_args,
      metadata_path=vc.metadata_path,
      custom_config=vc.get_vars()
      )
  )
