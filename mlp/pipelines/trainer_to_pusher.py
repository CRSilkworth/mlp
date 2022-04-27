"""The pipeline defintion which incrementally trains the intent classifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mlp.components.always_pusher import AlwaysPusher

from tfx.components import ImporterNode
from tfx.components import Trainer
from tfx.components.trainer.executor import GenericExecutor
from tfx.dsl.components.base import executor_spec
from tfx.proto import trainer_pb2
from tfx.proto import pusher_pb2
from tfx.types import standard_artifacts
from tfx.types import artifact
from tfx.types import artifact_utils
from tfx.orchestration import pipeline
from tfx.orchestration import metadata

from typing import Text, List, Dict, Optional, Any

import os


def create_pipeline(
  run_root: Text,
  pipeline_name: Text,
  pipeline_mod: Text,
  schema_uri: Text,
  transform_graph_uri: Text,
  examples_uri: Text,
  model_uri: Optional[Text] = None,
  beam_pipeline_args: Optional[List[Text]] = None,
  metadata_path: Optional[Text] = None,
  custom_config: Optional[Dict[Text, Any]] = None
) -> pipeline.Pipeline:
  """Implement the incremental pipeline from the Trainer component onward.

  Parameters
  ----------
  run_root: Path to the root directory to store all the output from the pipeline's run.
  pipeline_name: The name of the pipeline.
  pipeline_mod: The module of the pipeline (e.g. mod.submod.pipeline)
  schema_uri: The uri to the schema to import for this pipeline.
  transform_graph_uri: The uri to the transform graph to transform the raw data.
  transform_graph_uri: The uri to the examples to train on.
  beam_pipeline_args: The beam arguments for the entire pipeline.
  metadata_path: Where to store the metadata db.
  custom config: Any additional configuration to be passed to the Trainer.

  Returns
  -------
  pipeline: The tfx pipeline to be sent to beam/airflow/kubeflow etc.

  """
  components = []
  schema_importer = ImporterNode(
    instance_name='schema_importer',
    source_uri=schema_uri,
    artifact_type=standard_artifacts.Schema,
    reimport=False
  )
  components.append(schema_importer)

  transform_graph_importer = ImporterNode(
    instance_name='transform_graph_importer',
    source_uri=transform_graph_uri,
    artifact_type=standard_artifacts.TransformGraph,
    reimport=False
  )
  components.append(transform_graph_importer)

  examples_importer = ImporterNode(
    instance_name='examples_importer',
    source_uri=examples_uri,
    artifact_type=standard_artifacts.Examples,
    properties={
      'split_names':
      artifact_utils.encode_split_names(artifact.DEFAULT_EXAMPLE_SPLITS)},
    reimport=False
  )
  components.append(examples_importer)

  if model_uri is not None:
    model_importer = ImporterNode(
      instance_name='import_model',
      source_uri=model_uri,
      artifact_type=standard_artifacts.Model,
      reimport=False
    )
    components.append(model_importer)
    base_model = model_importer.outputs['result']
  else:
    base_model = None

  # Uses user-provided Python function that implements a model using TF-Learn.
  trainer = Trainer(
    transformed_examples=examples_importer.outputs['result'],
    custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
    schema=schema_importer.outputs['result'],
    transform_graph=transform_graph_importer.outputs['result'],
    train_args=trainer_pb2.TrainArgs(),
    eval_args=trainer_pb2.EvalArgs(),
    run_fn='{}.run_fn'.format(pipeline_mod),
    base_model=base_model,
    custom_config=custom_config
  )
  components.append(trainer)

  # Not depdent on blessing. Always pushes regardless of quality.
  pusher = AlwaysPusher(
    model=trainer.outputs['model'],
    push_destination=pusher_pb2.PushDestination(
      filesystem=pusher_pb2.PushDestination.Filesystem(
        base_directory=os.path.join(run_root, 'serving', 'model')
      )
    )
  )
  components.append(pusher)

  pipeline_kwargs = {}
  if metadata_path is not None:
    pipeline_kwargs = {
      'metadata_connection_config': metadata.sqlite_metadata_connection_config(
        metadata_path),
    }

  return pipeline.Pipeline(
    pipeline_name=pipeline_name,
    pipeline_root=os.path.join(run_root, 'data'),
    components=components,
    enable_cache=True,
    beam_pipeline_args=beam_pipeline_args,
    **pipeline_kwargs
  )
