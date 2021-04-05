"""The pipeline defintion which incrementally trains the intent classifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text, List, Dict, Optional, Any

from tfx.orchestration import pipeline
from tfx.orchestration import metadata

from tfx.components import ImporterNode
from mlp.components.transform_with_graph import TransformWithGraph
from tfx.components import Trainer
from tfx.extensions.google_cloud_big_query.example_gen.component import BigQueryExampleGen
from tfx.components.base import executor_spec
from mlp.components.always_pusher import AlwaysPusher
from mlp.components.artifact_pusher import SchemaPusher
from mlp.components.artifact_pusher import TransformGraphPusher

from tfx.proto import trainer_pb2
from tfx.proto import example_gen_pb2
from tfx.proto import pusher_pb2
from tfx.types import standard_artifacts

from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor

import os


def create_pipeline(
  prev_run_root: Text,
  run_root: Text,
  pipeline_name: Text,
  pipeline_mod: Text,
  query: Text,
  beam_pipeline_args: Optional[List[Text]] = None,
  metadata_path: Optional[Text] = None,
  custom_config: Optional[Dict[Text, Any]] = None
) -> pipeline.Pipeline:
  """Implements the incremental pipeline.."""

  example_gen = BigQueryExampleGen(
    query=query,
    output_config=example_gen_pb2.Output(
      split_config=example_gen_pb2.SplitConfig(
        splits=[
          example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=20),
          example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)]
      )
    )
  )

  schema_importer = ImporterNode(
    instance_name='import_schema',
    source_uri=os.path.join(prev_run_root, 'serving/schema'),
    artifact_type=standard_artifacts.Schema,
    reimport=False
  )
  graph_importer = ImporterNode(
    instance_name='import_transform_graph',
    source_uri=os.path.join(prev_run_root, 'serving/transform_graph'),
    artifact_type=standard_artifacts.TransformGraph,
    reimport=False
  )
  model_importer = ImporterNode(
    instance_name='import_model',
    source_uri=os.path.join(prev_run_root, 'serving/model'),
    artifact_type=standard_artifacts.Model,
    reimport=False
  )

  # Performs transformations and feature engineering in training and serving.
  transform = TransformWithGraph(
      examples=example_gen.outputs['examples'],
      schema=schema_importer.outputs['result'],
      transform_graph=graph_importer.outputs['result']
  )

  # Uses user-provided Python function that implements a model using TF-Learn.
  trainer = Trainer(
    transformed_examples=transform.outputs['transformed_examples'],
    schema=schema_importer.outputs['result'],
    transform_graph=graph_importer.outputs['result'],
    train_args=trainer_pb2.TrainArgs(),
    eval_args=trainer_pb2.EvalArgs(),
    trainer_fn='{}.trainer_fn'.format(pipeline_mod),
    base_model=model_importer.outputs['result'],
    custom_config=custom_config
  )

  # Not depdent on blessing. Always pushes regardless of quality.
  pusher = AlwaysPusher(
    model=trainer.outputs['model'],
    push_destination=pusher_pb2.PushDestination(
      filesystem=pusher_pb2.PushDestination.Filesystem(
        base_directory=os.path.join(run_root, 'serving', 'model')
      )
    )
  )

  schema_pusher = SchemaPusher(
      artifact=schema_importer.outputs['result'],
      push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
          base_directory=os.path.join(run_root, 'serving', 'schema')
        )
      ),
      instance_name='schema_pusher'
  )

  transform_graph_pusher = TransformGraphPusher(
      artifact=graph_importer.outputs['result'],
      push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
          base_directory=os.path.join(run_root, 'serving', 'transform_graph')
        )
      ),
      instance_name='transform_graph_pusher'
  )

  pipeline_kwargs = {}
  if metadata_path is not None:
    pipeline_kwargs = {
      'metadata_connection_config': metadata.sqlite_metadata_connection_config(
        metadata_path),
    }

  return pipeline.Pipeline(
    pipeline_name=pipeline_name,
    pipeline_root=os.path.join(run_root, 'data'),
    components=[
      example_gen,
      schema_importer,
      graph_importer,
      model_importer,
      transform,
      trainer,
      pusher,
      schema_pusher,
      transform_graph_pusher
    ],
    enable_cache=True,
    beam_pipeline_args=beam_pipeline_args,
    **pipeline_kwargs

  )
