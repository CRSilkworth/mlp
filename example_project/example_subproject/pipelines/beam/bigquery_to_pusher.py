"""Run a beam pipeline that starts from pulling examples from bigquery."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os

from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from mlp.pipelines.bigquery_to_pusher import create_pipeline

from __example_subproject__.preprocess import preprocess_factory
from __example_subproject__.train import trainer_factory
from mlp.utils.dirs import pipeline_dirs
from mlp.utils.dirs import pipeline_var_names
from mlp.utils.config import VarConfig

_MLP_PROJECT = '__example_project__'
_MLP_SUBPROJECT = '__example_subproject__'


_PIPELINE_TYPE = 'bigquery_to_pusher'
_RUNNER = 'beam'

# Set to timestamp of previous run if you want to continue old run.
_RUN_DIR = os.path.join(os.environ['HOME'], 'runs')

# Define the preprocessing/feature parameters
_CATEGORICAL_FEATURE_KEYS = ['vendor']
_NUMERICAL_FEATURE_KEYS = ['max_bottle_volume']
_LABEL_KEY = 'category'

trainer_fn = trainer_factory(
  categorical_feature_keys=_CATEGORICAL_FEATURE_KEYS,
  numerical_feature_keys=_NUMERICAL_FEATURE_KEYS,
  label_key=_LABEL_KEY,
)

preprocessing_fn = preprocess_factory(
  categorical_feature_keys=_CATEGORICAL_FEATURE_KEYS,
  numerical_feature_keys=_NUMERICAL_FEATURE_KEYS,
  label_key=_LABEL_KEY,
)

if __name__ == "__main__":
  vc = VarConfig()
  vc.gcp_project = '__gcp_project__'
  vc.mlp_project = _MLP_PROJECT
  vc.mlp_subproject = _MLP_SUBPROJECT

  vc.categorical_feature_keys = _CATEGORICAL_FEATURE_KEYS
  vc.numerical_feature_keys = _NUMERICAL_FEATURE_KEYS
  vc.label_key = _LABEL_KEY

  # Set to timestamp of previous run if you want to continue old run.
  vc.runner = _RUNNER
  vc.run_dir = _RUN_DIR
  vc.pipeline_type = _PIPELINE_TYPE
  vc.run_str = None

  vc.query = """
    SELECT
      item_description,
      MAX(vendor_name) AS vendor,
      MAX(bottle_volume_ml) AS max_bottle_volume,
      MAX(category_name) AS category
    FROM `bigquery-public-data.iowa_liquor_sales.sales`
    GROUP BY item_description
  """

  # Define the training/model parameters
  vc.hidden_layer_dims = [10]
  vc.batch_size = 32
  vc.num_train_steps = 1000
  vc.num_eval_steps = 100
  vc.warmup_prop = 0.1
  vc.cooldown_prop = 0.1
  vc.warm_start_from = None
  vc.save_summary_steps = 100
  vc.save_checkpoints_secs = 3600
  vc.learning_rate = 2e-5
  vc.num_gpus = 1

  vc.add_vars(**pipeline_var_names(
      vc.run_dir,
      vc.run_str,
      vc.mlp_project,
      vc.mlp_subproject,
      vc.runner,
      vc.pipeline_type
    )
  )

  vc.beam_pipeline_args = [
    '--project=' + vc.gcp_project,
    '--temp_location=' + os.path.join('gs://__gcp_bucket__', 'tmp'),
  ]

  vc.write(vc.vc_config_path)
  DAG = BeamDagRunner().run(
    create_pipeline(
      run_root=vc.run_root,
      pipeline_name=vc.pipeline_name,
      pipeline_mod=vc.pipeline_mod,
      query=vc.query,
      beam_pipeline_args=vc.beam_pipeline_args,
      metadata_path=os.path.join(vc.run_root, 'metadata', 'metadata.db'),
      custom_config=vc.get_vars()
      )
  )
