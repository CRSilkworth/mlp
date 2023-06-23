"""Create a kubeflow pipeline that starts pulling examples from bigquery."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import sys

from mlp.pipelines.bigquery_to_pusher import create_pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner

from __example_subproject__ import preprocess
from __example_subproject__ import train
from mlp.utils.dirs import pipeline_dirs
from mlp.kubeflow.pipeline_ops import set_gpu_limit
from mlp.utils.dirs import pipeline_var_names
from mlp.utils.sql import query_with_kwargs
from mlp.utils.config import VarConfig

_MLP_PROJECT = '__example_project__'
_MLP_SUBPROJECT = '__example_subproject__'

_PIPELINE_TYPE = 'bigquery_to_pusher'
_RUNNER = 'kubeflow'

# Set to timestamp of previous run if you want to continue old run.
_RUN_DIR = os.path.join('gs://__gcp_bucket__', 'runs')

# Define the preprocessing/feature parameters
_CATEGORICAL_FEATURE_KEYS = ['vendor']
_NUMERICAL_FEATURE_KEYS = ['max_bottle_volume']
_LABEL_KEY = 'category'

run_fn = train.trainer_factory(
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
  vc.gcp_project = '__gcp_project__'
  vc.gcp_region = '__gcp_region__'
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
  if len(sys.argv) > 1:
    vc.run_str = sys.argv[1]

  vc.experiment = None
  if len(sys.argv) > 2:
    vc.experiment = sys.argv[2]

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
  vc.steps_per_epoch = 200
  vc.num_eval_steps = 20
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
      vc.pipeline_type,
      vc.experiment
    )
  )

  vc.image_name = 'gcr.io/{gcp_project}/{mlp_project}'.format(
    gcp_project=vc.gcp_project,
    mlp_project=vc.mlp_project,
  )

  # If running with dataflow
  vc.beam_pipeline_args = [
    '--experiments=shuffle_mode=auto',
    '--project=' + vc.gcp_project,
    '--temp_location=' + os.path.join(vc.run_dir, 'tmp'),
    '--region=' + vc.gcp_region,
    # If you want to use DataFlow, ensure that the service account
    # <kf-deployment-name>-user@<gcp_project>.iam.gserviceaccount.com has the
    # ServiceAccount/ServiceAccountUser role. Then uncomment everything below.
    # '--runner=DataflowRunner',
    # # Need to explicitly pass the image, otherwise dataflow doesn't work for
    # # transform. It can't find any usercode (e.g. mlp)
    # '--experiment=use_runner_v2',
    # '--worker_harness_container_image=gcr.io/{gcp_project}/{mlp_project}:{version}'.format(
    #   gcp_project=vc.gcp_project,
    #   mlp_project=vc.mlp_project,
    #   version=vc.version
    # ),
    # # Need 50GB otherwise dataflow hangs without any errors
    # '--disk_size_gb=50',
    # '--machine_type=n1-standard-4',
  ]

  pipeline_op_funcs = kubeflow_dag_runner.get_default_pipeline_operator_funcs()
  if vc.num_gpus:
    pipeline_op_funcs.append(set_gpu_limit(vc.num_gpus))

  runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
    kubeflow_metadata_config=kubeflow_dag_runner.get_default_kubeflow_metadata_config(),
    pipeline_operator_funcs=pipeline_op_funcs,
    # tfx_image=os.environ.get('KUBEFLOW_TFX_IMAGE', None)
    tfx_image=vc.image_name
  )

  vc.hash = vc.get_hash()
  vc.write(vc.vc_config_path)
  kubeflow_dag_runner.KubeflowDagRunner(config=runner_config).run(
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
