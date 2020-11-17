"""Run a beam pipeline that starts from pulling examples from bigquery."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os

from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from example_subproject.pipelines.defs.bigquery_to_pusher import create_pipeline

from example_subproject import preprocess
from example_subproject import train
from mlp.utils.dirs import pipeline_dirs
from mlp.utils.dirs import pipeline_var_names
from mlp.utils.config import VarConfig

_MLP_PROJECT = 'example_project'
_MLP_SUBPROJECT = 'example_subproject'


_PIPELINE_TYPE = 'bigquery_to_pusher'
_RUNNER = 'beam'

# Set to timestamp of previous run if you want to continue old run.
_RUN_DIR = os.path.join(os.environ['HOME'], 'runs')

# Define the preprocessing/feature parameters
_CATEGORICAL_FEATURE_KEYS = ['vendor']
_NUMERICAL_FEATURE_KEYS = ['max_bottle_volume']
_LABEL_KEY = 'category'

trainer_fn = train.trainer_factory(
  categorical_feature_keys=_CATEGORICAL_FEATURE_KEYS,
  numerical_feature_keys=_NUMERICAL_FEATURE_KEYS,
  label_key=_LABEL_KEY,
)

preprocessing_fn = preprocess.preprocess_factory(
  categorical_feature_keys=_CATEGORICAL_FEATURE_KEYS,
  numerical_feature_keys=_NUMERICAL_FEATURE_KEYS,
  label_key=_LABEL_KEY,
)

if __name__ == "__main__":
  vc = VarConfig()
  vc.gcp_project = 'gcp_project'
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

  _QUERY = """
    SELECT
      item_description,
      MAX(vendor_name) AS vendor,
      MAX(bottle_volume_ml) AS max_bottle_volume,
      MAX(category_name) AS category
    FROM `bigquery-public-data.iowa_liquor_sales.sales`
    GROUP BY item_description
  """

  # Define the training/model parameters
  _HIDDEN_LAYER_DIMS = [10]
  _BATCH_SIZE = 32
  _NUM_TRAIN_STEPS = 1000
  _NUM_EVAL_STEPS = 100
  _WARMUP_PROP = 0.1
  _COOLDOWN_PROP = 0.1
  _WARM_START_FROM = None
  _SAVE_SUMMARY_STEPS = 100
  _SAVE_CHECKPOINT_SECS = 3600
  _LEARNING_RATE = 2e-5
  _NUM_GPUS = 1

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
    '--project=' + vc.gcp_project
  ]

  vc.write(vc.vc_config_path)
  DAG = BeamDagRunner().run(
    create_pipeline(
      run_root=vc.run_root,
      pipeline_root=vc.pipeline_root,
      pipeline_mod=vc.pipeline_mod,
      query=vc.query,
      beam_pipeline_args=vc.beam_pipeline_args,
      metadata_path=os.path.join(vc.run_root, 'metadata', 'metadata.db'),
      custom_config=vc.get_vars()
      )
  )
