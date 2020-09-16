"""Create a kubeflow pipeline that starts pulling examples from bigquery."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os

from example_subproject.pipelines.defs.bigquery_to_pusher import create_pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner

from example_subproject import preprocess
from example_subproject import train
from mlp.utils.dirs import pipeline_dirs
from mlp.kubeflow.pipeline_ops import set_gpu_limit

_MLP_PROJECT = 'example_project'
_MLP_SUBPROJECT = 'example_subproject'

_PIPELINE_TYPE = 'bigquery_to_pusher'
_RUNNER = 'kubeflow'

# Set to timestamp of previous run if you want to continue old run.
_RUN_STR = None
_RUN_DIR = os.path.join('gs://gcp_bucket')

_GCP_PROJECT = 'gcp_project'
_GCP_REGION = 'gcp_region'

_QUERY = """
  SELECT
    item_description,
    MAX(vendor_name) AS vendor,
    MAX(bottle_volume_ml) AS max_bottle_volume,
    MAX(category_name) AS category
  FROM `bigquery-public-data.iowa_liquor_sales.sales`
  GROUP BY item_description
"""

# Define the preprocessing/feature parameters
_CATEGORICAL_FEATURE_KEYS = ['vendor']
_NUMERICAL_FEATURE_KEYS = ['max_bottle_volume']
_LABEL_KEY = 'category'

# Define the training/model parameters
_HIDDEN_LAYER_DIMS = [10]
_BATCH_SIZE = 32
_NUM_TRAIN_STEPS = 5000000
_NUM_EVAL_STEPS = 10000
_WARMUP_PROP = 0.1
_COOLDOWN_PROP = 0.1
_WARM_START_FROM = None
_SAVE_SUMMARY_STEPS = 100
_SAVE_CHECKPOINT_SECS = 3600
_LEARNING_RATE = 2e-5
_NUM_GPUS = 1

pipeline_name = '-'.join([
  _MLP_PROJECT,
  _MLP_SUBPROJECT,
  _PIPELINE_TYPE
])
pipeline_mod = '.'.join([
  _MLP_SUBPROJECT,
  'pipelines',
  _RUNNER,
  _PIPELINE_TYPE
])
proj_root, run_root, pipeline_root, serving_uri = pipeline_dirs(
  _RUN_DIR,
  _RUN_STR,
  _MLP_PROJECT,
  _MLP_SUBPROJECT,
  pipeline_name
)

trainer_fn = train.trainer_factory(
  batch_size=_BATCH_SIZE,
  learning_rate=_LEARNING_RATE,
  hidden_layer_dims=_HIDDEN_LAYER_DIMS,
  categorical_feature_keys=_CATEGORICAL_FEATURE_KEYS,
  numerical_feature_keys=_NUMERICAL_FEATURE_KEYS,
  label_key=_LABEL_KEY,
  warmup_prop=_WARMUP_PROP,
  cooldown_prop=_COOLDOWN_PROP,
  # warm_start_from=_WARM_START_FROM,
  save_summary_steps=_SAVE_SUMMARY_STEPS,
  save_checkpoints_secs=_SAVE_CHECKPOINT_SECS
)

preprocessing_fn = preprocess.preprocess_factory(
  categorical_feature_keys=_CATEGORICAL_FEATURE_KEYS,
  numerical_feature_keys=_NUMERICAL_FEATURE_KEYS,
  label_key=_LABEL_KEY,
)

# If running with dataflow
beam_pipeline_args = [
  # If you want to use DataFlow, ensure that the service account
  # <kf-deployment-name>-user@<gcp_project>.iam.gserviceaccount.com has the
  # ServiceAccount/ServiceAccountUser role.
  # '--runner=DataflowRunner',
  '--experiments=shuffle_mode=auto',
  '--project=' + _GCP_PROJECT,
  '--temp_location=' + os.path.join(_RUN_DIR, 'tmp'),
  '--region=' + _GCP_REGION,
  '--disk_size_gb=50',
]
pipeline_op_funcs = kubeflow_dag_runner.get_default_pipeline_operator_funcs()

if _NUM_GPUS:
  pipeline_op_funcs.append(set_gpu_limit(_NUM_GPUS))

if __name__ == "__main__":
  runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
    kubeflow_metadata_config=kubeflow_dag_runner.get_default_kubeflow_metadata_config(),
    pipeline_operator_funcs=pipeline_op_funcs,
    tfx_image=os.environ.get('KUBEFLOW_TFX_IMAGE', None)
  )

  kubeflow_dag_runner.KubeflowDagRunner(config=runner_config).run(
    create_pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      pipeline_mod=pipeline_mod,
      query=_QUERY,
      serving_uri=serving_uri,
      num_train_steps=_NUM_TRAIN_STEPS,
      num_eval_steps=_NUM_EVAL_STEPS,
      beam_pipeline_args=beam_pipeline_args,
      )
  )
