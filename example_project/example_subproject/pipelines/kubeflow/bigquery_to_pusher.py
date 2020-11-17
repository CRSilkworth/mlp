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
from mlp.utils.dirs import pipeline_var_names
from mlp.utils.sql import query_with_kwargs
from mlp.utils.config import VarConfig

_MLP_PROJECT = 'example_project'
_MLP_SUBPROJECT = 'example_subproject'

_PIPELINE_TYPE = 'bigquery_to_pusher'
_RUNNER = 'kubeflow'

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
  vc.gcp_region = 'gcp_region'
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
  vc.hidden_layer_dims = [10]
  vc.batch_size = 32
  vc.num_train_steps = 5000000
  vc.num_eval_steps = 10000
  vc.warmup_prop = 0.1
  vc.cooldown_prop = 0.1
  vc.warm_start_from = None
  vc.save_summary_steps = 100
  vc.save_checkpoint_secs = 3600
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

  # If running with dataflow
  vc.beam_pipeline_args = [
    # If you want to use DataFlow, ensure that the service account
    # <kf-deployment-name>-user@<gcp_project>.iam.gserviceaccount.com has the
    # ServiceAccount/ServiceAccountUser role.
    # '--runner=DataflowRunner',
    '--experiments=shuffle_mode=auto',
    '--project=' + vc.gcp_project,
    '--temp_location=' + os.path.join(vc.run_dir, 'tmp'),
    '--region=' + vc.gcp_region,
    '--disk_size_gb=50',
  ]

  pipeline_op_funcs = kubeflow_dag_runner.get_default_pipeline_operator_funcs()
  if vc.num_gpus:
    pipeline_op_funcs.append(set_gpu_limit(vc.num_gpus))

  runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
    kubeflow_metadata_config=kubeflow_dag_runner.get_default_kubeflow_metadata_config(),
    pipeline_operator_funcs=pipeline_op_funcs,
    tfx_image=os.environ.get('KUBEFLOW_TFX_IMAGE', None)
  )

  vc.write(vc.vc_config_path)
  kubeflow_dag_runner.KubeflowDagRunner(config=runner_config).run(
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
