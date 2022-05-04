"""Create a kubeflow pipeline that starts pulling examples from bigquery."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from mlp.pipelines.bigquery_to_pusher import create_pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner

from __example_subproject__ import preprocess
from __example_subproject__ import train
import __example_subproject__.pipelines.kubflow.bigquery_to_pusher as full
from mlp.utils.dir import pipeline_dirs
from mlp.utils.resolvers import multi_pipeline_uri
from mlp.utils.resolvers import latest_run_root
from mlp.utils.resolvers import latest_artifact_path
from mlp.utils.config import VarConfig
from mlp.utils.dirs import pipeline_var_names

_PIPELINE_TYPE = 'incremental'

trainer_fn = full.trainer_fn

if __name__ == "__main__":
  prev_run_root = latest_run_root(
    full._RUN_DIR,
    full._MLP_PROJECT,
    full._MLP_SUBPROJECT,
    full._PIPELINE_TYPE,
    _PIPELINE_TYPE
  )

  vc = VarConfig(os.path.join(prev_run_root, 'config', 'pipeline_vars.json'))
  vc.prev_run_root = prev_run_root
  vc.pipeline_type = _PIPELINE_TYPE
  vc.run_str = None

  vc.freq_num = -3
  vc.freq = "month"
  vc.num_old = 5000
  vc.query = """
  (
    SELECT
      item_description,
      MAX(vendor_name) AS vendor,
      MAX(bottle_volume_ml) AS max_bottle_volume,
      MAX(category_name) AS category
    FROM `bigquery-public-data.iowa_liquor_sales.sales`
    WHERE date > DATE_ADD(CURRENT_DATE(), INTERVAL {} {})
    GROUP BY item_description
  )
  UNION DISTINCT
  (
    SELECT
      item_description,
      MAX(vendor_name) AS vendor,
      MAX(bottle_volume_ml) AS max_bottle_volume,
      MAX(category_name) AS category
    FROM `bigquery-public-data.iowa_liquor_sales.sales`
    GROUP BY item_description
    LIMIT {}
   )
  """.format(vc.freq_num, vc.freq, vc.num_old)

  # Define the training/model parameters
  vc.num_train_steps = 100
  vc.num_eval_steps = 10
  vc.warmup_prop = 0.1
  vc.cooldown_prop = 0.1
  vc.save_summary_steps = 10
  vc.save_checkpoints_secs = 3600
  vc.learning_rate = 2e-5

  runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
    kubeflow_metadata_config=kubeflow_dag_runner.get_default_kubeflow_metadata_config(),
    # tfx_image=os.environ.get('KUBEFLOW_TFX_IMAGE', None)
    tfx=vc.image_name
  )

  kubeflow_dag_runner.KubeflowDagRunner(config=runner_config).run(
    create_pipeline(
      prev_run_root=vc.prev_run_root,
      run_root=vc.run_root,
      pipeline_name=vc.pipeline_name,
      pipeline_mod=vc.pipeline_mod,
      query=vc.query,
      base_model_uri=vc.base_model_uri,
      beam_pipeline_args=vc.beam_pipeline_args,
      metadata_path=vc.metadata_path,
      custom_config=vc.get_vars(),
      )
  )
