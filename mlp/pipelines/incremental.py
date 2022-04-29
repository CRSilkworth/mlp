"""The pipeline defintion which incrementally trains the intent classifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mlp.components.always_pusher import AlwaysPusher
from mlp.components.artifact_pusher import SchemaPusher
from mlp.components.artifact_pusher import TransformGraphPusher
from mlp.components.transform_with_graph import TransformWithGraph

from tfx.dsl.components.common import Importer
from tfx.components import Trainer
from tfx.components import BigQueryExampleGen
from tfx.orchestration import pipeline
from tfx.orchestration import metadata
from tfx.proto import trainer_pb2
from tfx.proto import example_gen_pb2
from tfx.proto import pusher_pb2
from tfx.types import standard_artifacts

from typing import Text, List, Dict, Optional, Any

import os


def create_pipeline(
  prev_run_root: Text,
  run_root: Text,
  pipeline_name: Text,
  pipeline_mod: Text,
  query: Text,
  beam_pipeline_args: Optional[List[Text]] = None,
  metadata_path: Optional[Text] = None,
  custom_config: Optional[Dict[Text, Any]] = None,
  train_hash_buckets: Optional[int] = 19,
  eval_hash_buckets: Optional[int] = 1,
) -> pipeline.Pipeline:
  """Implement an pipeline to incrementally train a model.

  Parameters
  ----------
  prev_run_root: The path to the root of the run directory that has all the artifacts needed to run an incremental training (e.g. transform graph, schema, model)
  run_root: Path to the root directory to store all the output from the pipeline's run.
  pipeline_name: The name of the pipeline.
  pipeline_mod: The module of the pipeline (e.g. mod.submod.pipeline)
  query: The query to be run on bigquery to generate the examples.
  beam_pipeline_args: The beam arguments for the entire pipeline.
  metadata_path: Where to store the metadata db.
  custom config: Any additional configuration to be passed to the Transform/Trainer.
  train_hash_buckets: The proportion of examples to be used to train with.
  eval_hash_buckets: The proportion of examples to be used to eval with.


  Returns
  -------
  pipeline: The tfx pipeline to be sent to beam/airflow/kubeflow etc.

  """
  example_gen = BigQueryExampleGen(
    query=query,
    output_config=example_gen_pb2.Output(
      split_config=example_gen_pb2.SplitConfig(
        splits=[
          example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=train_hash_buckets),
          example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=eval_hash_buckets)]
      )
    )
  )

  schema_importer = Importer(
    source_uri=os.path.join(prev_run_root, 'serving/schema'),
    artifact_type=standard_artifacts.Schema,
    reimport=False
  ).with_id('schema_importer')

  transform_graph_importer = Importer(
    source_uri=os.path.join(prev_run_root, 'serving/transform_graph'),
    artifact_type=standard_artifacts.TransformGraph,
    reimport=False
  ).with_id('transform_graph_importer')

  model_importer = Importer(
    source_uri=os.path.join(prev_run_root, 'serving/model'),
    artifact_type=standard_artifacts.Model,
    reimport=False
  ).with_id('model_importer')

  # Performs transformations and feature engineering in training and serving.
  transform = TransformWithGraph(
      examples=example_gen.outputs['transformed_examples'],
      schema=schema_importer.outputs['result'],
      transform_graph=transform_graph_importer.outputs['result']
  )

  # Uses user-provided Python function that implements a model using TF-Learn.
  trainer = Trainer(
    examples=transform.outputs['transformed_examples'],
    schema=schema_importer.outputs['result'],
    transform_graph=transform_graph_importer.outputs['result'],
    train_args=trainer_pb2.TrainArgs(),
    eval_args=trainer_pb2.EvalArgs(),
    run_fn='{}.run_fn'.format(pipeline_mod),
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
  ).with_id('schema_pusher')

  transform_graph_pusher = TransformGraphPusher(
      artifact=transform_graph_importer.outputs['result'],
      push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
          base_directory=os.path.join(run_root, 'serving', 'transform_graph')
        )
      ),
  ).with_id('transform_graph_pusher')

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
      transform_graph_importer,
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
