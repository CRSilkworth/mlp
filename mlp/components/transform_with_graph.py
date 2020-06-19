from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Text, Any, Dict, List, Mapping, Tuple, Iterable, Sequence, Union, Generator

import absl
import os
import tensorflow as tf
import pandas as pd

from tfx import types
from tfx.components.base import base_component
from tfx.components.base import executor_spec

from tfx.types import standard_artifacts
from tfx.types import artifact
from tfx.types import artifact_utils

from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ExecutionParameter
from tfx.components.base import base_executor
from tfx.components.util import value_utils
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import dataset_metadata
from tfx.utils import import_utils
from tensorflow.python.lib.io import file_io
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_transform.beam import analyzer_cache
from google.protobuf import json_format
import apache_beam as beam
from tfx.proto import example_gen_pb2
import datetime
import numpy as np
from tfx.utils import io_utils

EXAMPLES_KEY = 'examples'
# Key for schema in executor input_dict.
SCHEMA_KEY = 'schema'

# Key for temp path, for internal use only.
TEMP_PATH_KEY = 'temp_path'

# Key for transform graph in executor output_dict.
TRANSFORM_GRAPH_KEY = 'transform_graph'

# Key for output model in executor output_dict.
TRANSFORMED_EXAMPLES_KEY = 'transformed_examples'

_DEFAULT_TRANSFORMED_EXAMPLES_PREFIX = 'transformed_examples'

_TRANSFORM_INTERNAL_FEATURE_FOR_KEY = '__TFT_PASS_KEY__'

_TEMP_DIR_IN_TRANSFORM_OUTPUT = '.temp_path'


class _Dataset(object):
  """Dataset to be analyzed and/or transformed.

  It also contains bundle of stages of a single dataset through the transform
  pipeline.
  """
  _FILE_PATTERN_SUFFIX_LENGTH = 6

  def __init__(self, file_pattern: Text,
               materialize_output_path: Optional[Text] = None):
    """Initialize a Dataset.

    Args:
      file_pattern: The file pattern of the dataset.
      materialize_output_path: The file path where to write the dataset.
    """
    file_pattern_suffix = os.path.join(
        *file_pattern.split(os.sep)[-self._FILE_PATTERN_SUFFIX_LENGTH:])
    self._file_pattern = file_pattern
    self._materialize_output_path = materialize_output_path
    self._index = None
    self._serialized = None
    self._decoded = None
    self._transformed = None
    self._transformed_and_serialized = None
    if hasattr(analyzer_cache, 'DatasetKey'):
      self._dataset_key = analyzer_cache.DatasetKey(file_pattern_suffix)
    else:
      self._dataset_key = analyzer_cache.make_dataset_key(file_pattern_suffix)
    print('-'*50)
    print(self._dataset_key)
    print('-'*50)
  @property
  def file_pattern(self):
    assert self._file_pattern
    return self._file_pattern

  @property
  def materialize_output_path(self):
    assert self._materialize_output_path
    return self._materialize_output_path

  @property
  def index(self):
    assert self._index is not None
    return self._index

  @property
  def dataset_key(self):
    assert self._dataset_key
    return self._dataset_key

  @property
  def serialized(self):
    assert self._serialized is not None
    return self._serialized

  @property
  def decoded(self):
    assert self._decoded is not None
    return self._decoded

  @property
  def transformed(self):
    assert self._transformed is not None
    return self._transformed

  @property
  def transformed_and_serialized(self):
    assert self._transformed_and_serialized is not None
    return self._transformed_and_serialized

  @index.setter
  def index(self, val):
    self._index = val

  @serialized.setter
  def serialized(self, val):
    self._serialized = val

  @decoded.setter
  def decoded(self, val):
    self._decoded = val

  @transformed.setter
  def transformed(self, val):
    self._transformed = val

  @transformed_and_serialized.setter
  def transformed_and_serialized(self, val):
    self._transformed_and_serialized = val

def _GetSchemaProto(
    metadata: dataset_metadata.DatasetMetadata) -> schema_pb2.Schema:
  """Gets the schema proto associated with a DatasetMetadata.

  This is needed because tensorflow_transform 0.13 and tensorflow_transform 0.14
  have a different API for DatasetMetadata.

  Args:
    metadata: A dataset_metadata.DatasetMetadata.

  Returns:
    A schema_pb2.Schema.
  """
  # `schema` is either a Schema proto or dataset_schema.Schema.
  schema = metadata.schema
  # In the case where it's a dataset_schema.Schema, fetch the schema proto.
  return getattr(schema, '_schema_proto', schema)

class Executor(base_executor.BaseExecutor):
  """Executor for Slack component."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Get human review result on a model through Slack channel.

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - model_export: exported model from trainer.
        - model_blessing: model blessing path from model_validator.
      output_dict: Output dict from key to a list of artifacts, including:
        - slack_blessing: model blessing result.
      exec_properties: A dict of execution properties, including:
        - slack_token: Token used to setup connection with slack server.
        - slack_channel_id: The id of the Slack channel to send and receive
          messages.
        - timeout_sec: How long do we wait for response, in seconds.

    Returns:
      None

    Raises:
      TimeoutError:
        When there is no decision made within timeout_sec.
      ConnectionError:
        When connection to slack server cannot be established.

    """
    self._log_startup(input_dict, output_dict, exec_properties)
    transform_graph_uri = artifact_utils.get_single_uri(
        input_dict[TRANSFORM_GRAPH_KEY])
    temp_path = os.path.join(transform_graph_uri, _TEMP_DIR_IN_TRANSFORM_OUTPUT)
    # transformed_schema_file = os.path.join(
    #   transform_graph_uri,
    #   tft.TFTransformOutput.TRANSFORMED_METADATA_DIR,
    #   'schema.pbtxt'
    # )
    # transformed_schema_proto = io_utils.parse_pbtxt_file(
    #   transformed_schema_file,
    #   schema_pb2.Schema()
    # )
    transformed_train_output = artifact_utils.get_split_uri(
      output_dict[TRANSFORMED_EXAMPLES_KEY], 'train')
    transformed_eval_output = artifact_utils.get_split_uri(
      output_dict[TRANSFORMED_EXAMPLES_KEY], 'eval')

    tf_transform_output = tft.TFTransformOutput(transform_graph_uri)
    # transform_output_dataset_metadata = dataset_metadata.DatasetMetadata(
    #   schema=transformed_schema_proto
    # )

    # transform_fn = (tf_transform_output.transform_raw_features, transform_output_dataset_metadata)
    # feature_spec = schema_utils.schema_as_feature_spec(schema_proto).feature_spec
    schema_file = io_utils.get_only_uri_in_dir(
        artifact_utils.get_single_uri(input_dict[SCHEMA_KEY]))
    schema_proto = io_utils.parse_pbtxt_file(schema_file, schema_pb2.Schema())
    transform_input_dataset_metadata = dataset_metadata.DatasetMetadata(
      schema_proto
    )

    train_data_uri = artifact_utils.get_split_uri(
      input_dict[EXAMPLES_KEY],
      'train'
    )
    eval_data_uri = artifact_utils.get_split_uri(
      input_dict[EXAMPLES_KEY],
      'eval'
    )
    analyze_data_paths = [io_utils.all_files_pattern(train_data_uri)]
    transform_data_paths = [
      io_utils.all_files_pattern(train_data_uri),
      io_utils.all_files_pattern(eval_data_uri),
    ]
    materialize_output_paths = [
      os.path.join(transformed_train_output, _DEFAULT_TRANSFORMED_EXAMPLES_PREFIX),
      os.path.join(transformed_eval_output, _DEFAULT_TRANSFORMED_EXAMPLES_PREFIX)
    ]
    transform_data_list = self._MakeDatasetList(
      transform_data_paths,
      materialize_output_paths
    )
    analyze_data_list = self._MakeDatasetList(
      analyze_data_paths,
    )

    with self._make_beam_pipeline() as pipeline:
      with tft_beam.Context(temp_dir=temp_path):
        # NOTE: Unclear if there is a difference between input_dataset_metadata
        # and transform_input_dataset_metadata. Look at Transform executor.
        decode_fn = tft.coders.ExampleProtoCoder(schema_proto, serialized=True).decode

        input_analysis_data = {}
        for dataset in analyze_data_list:
          infix = 'AnalysisIndex{}'.format(dataset.index)
          dataset.serialized = (
            pipeline
            | 'ReadDataset[{}]'.format(infix) >> self._ReadExamples(
                dataset, transform_input_dataset_metadata))
          dataset.decoded = (
            dataset.serialized
            | 'Decode[{}]'.format(infix)
            >> self._DecodeInputs(decode_fn))
          input_analysis_data[dataset.dataset_key] = dataset.decoded

        if not hasattr(tft_beam.analyzer_cache, 'DatasetKey'):
          input_analysis_data = (
              [
                  dataset for dataset in input_analysis_data.values()
                  if dataset is not None
              ]
              | 'FlattenAnalysisDatasetsBecauseItIsRequired' >>
              beam.Flatten(pipeline=pipeline))

        transform_fn = (
            (input_analysis_data, transform_input_dataset_metadata)
            | 'Analyze' >> tft_beam.AnalyzeDataset(
                tf_transform_output.transform_raw_features, pipeline=pipeline))

        for dataset in transform_data_list:
          infix = 'TransformIndex{}'.format(dataset.index)
          dataset.serialized = (
            pipeline
            | 'ReadDataset[{}]'.format(infix) >> self._ReadExamples(
                dataset, transform_input_dataset_metadata))

          dataset.decoded = (
            dataset.serialized
            | 'Decode[{}]'.format(infix)
            >> self._DecodeInputs(decode_fn))

          dataset.transformed, metadata = (
              ((dataset.decoded, transform_input_dataset_metadata), transform_fn)
              | 'Transform[{}]'.format(infix) >> tft_beam.TransformDataset())

          dataset.transformed_and_serialized = (
              dataset.transformed
              | 'EncodeAndSerialize[{}]'.format(infix)
              >> beam.ParDo(self._EncodeAsSerializedExamples(), _GetSchemaProto(metadata)))

          _ = (
            dataset.transformed_and_serialized
            | 'Materialize[{}]'.format(infix) >> self._WriteExamples(dataset.materialize_output_path))

  @staticmethod
  @beam.ptransform_fn
  @beam.typehints.with_input_types(beam.Pipeline)
  @beam.typehints.with_output_types(Tuple[bytes, bytes])
  def _ReadExamples(
      pipeline: beam.Pipeline,
      dataset: _Dataset,
      input_dataset_metadata: dataset_metadata.DatasetMetadata
  ) -> beam.pvalue.PCollection:
    """Reads examples from the given `dataset`.

    Args:
      pipeline: beam pipeline.
      dataset: A `_Dataset` object that represents the data to read.
      input_dataset_metadata: A `dataset_metadata.DatasetMetadata`. Not used.

    Returns:
      A PCollection containing KV pairs of bytes.
    """
    del input_dataset_metadata
    return (
        pipeline
        | 'Read' >> beam.io.ReadFromTFRecord(
            dataset.file_pattern,
            coder=beam.coders.BytesCoder(),
            # TODO(b/114938612): Eventually remove this override.
            validate=False)
        | 'AddKey' >> beam.Map(lambda x: (None, x)))

  @staticmethod
  @beam.ptransform_fn
  @beam.typehints.with_input_types(Tuple[bytes, bytes])
  @beam.typehints.with_output_types(Dict[Text, Any])
  def _DecodeInputs(pcoll: beam.pvalue.PCollection,
                    decode_fn: Any) -> beam.pvalue.PCollection:
    """Decodes the given PCollection while handling KV data.

    Args:
      pcoll: PCollection of data.
      decode_fn: Function used to decode data.

    Returns:
      PCollection of decoded data.
    """

    def decode_example(kv: Tuple[Optional[bytes], bytes]) -> Dict[Text, Any]:  # pylint: disable=invalid-name
      """Decodes a single example."""
      (key, value) = kv
      result = decode_fn(value)
      if _TRANSFORM_INTERNAL_FEATURE_FOR_KEY in result:
        raise ValueError('"{}" is a reserved feature name, '
                         'it should not be present in the dataset.'.format(
                             _TRANSFORM_INTERNAL_FEATURE_FOR_KEY))
      result[_TRANSFORM_INTERNAL_FEATURE_FOR_KEY] = key
      return result

    return pcoll | 'ApplyDecodeFn' >> beam.Map(decode_example)

  @staticmethod
  @beam.ptransform_fn
  @beam.typehints.with_input_types(Tuple[bytes, bytes])
  @beam.typehints.with_output_types(beam.pvalue.PDone)
  def _WriteExamples(pcoll: beam.pvalue.PCollection,
                     transformed_example_path: Text) -> beam.pvalue.PDone:
    """Writes transformed examples compressed in gzip format.

    Args:
      pcoll: PCollection of serialized transformed examples.
      transformed_example_path: path to write to.

    Returns:
      beam.pvalue.PDone.
    """
    return (
        pcoll
        | 'Values' >> beam.Values()
        | 'Write' >> beam.io.WriteToTFRecord(
            transformed_example_path, file_name_suffix='.gz'))

  def _MakeDatasetList(
      self,
      file_patterns: Sequence[Union[Text, int]],
      materialize_output_paths: Optional[Sequence[Text]] = None
  ) -> List[_Dataset]:
    """Makes a list of Dataset from the given `file_patterns`.

    Args:
      file_patterns: A list of file patterns where each pattern corresponds to
        one `_Dataset`.
      can_process_jointly: Whether paths can be processed jointly, unused.
      materialize_output_paths: The materialization output paths, if applicable.

    Returns:
      A list of `_Dataset` sorted by their dataset_key property.
    """
    if materialize_output_paths:
      assert len(file_patterns) == len(materialize_output_paths)
    else:
      materialize_output_paths = [None] * len(file_patterns)

    datasets = [
      _Dataset(p, m)
      for p, m in zip(file_patterns, materialize_output_paths)
    ]
    result = sorted(datasets, key=lambda dataset: dataset.dataset_key)
    for index, dataset in enumerate(result):
      dataset.index = index
    return result

  @beam.typehints.with_input_types(Dict[Text, Any], schema=schema_pb2.Schema)
  @beam.typehints.with_output_types(Tuple[Optional[bytes], bytes])
  class _EncodeAsSerializedExamples(beam.DoFn):
    """Encodes data as serialized tf.Examples based on the given metadata."""

    def __init__(self):
      self._coder = None

    def process(
      self,
      element: Dict[Text, Any],
      schema: schema_pb2.Schema
      ) -> Generator[Tuple[Any, Any], None, None]:
      if self._coder is None:
        self._coder = tft.coders.ExampleProtoCoder(schema, serialized=True)

      # Make sure that the synthetic key feature doesn't get encoded.
      key = element.get(_TRANSFORM_INTERNAL_FEATURE_FOR_KEY, None)
      if key is not None:
        element = element.copy()
        del element[_TRANSFORM_INTERNAL_FEATURE_FOR_KEY]
      yield (key, self._coder.encode(element))


class TransformWithGraphSpec(types.ComponentSpec):
  """ComponentSpec for Custom TFX Slack Component."""

  PARAMETERS = {}
  INPUTS = {
    'examples': ChannelParameter(type=standard_artifacts.Examples),
    'schema': ChannelParameter(type=standard_artifacts.Schema),
    'transform_graph': ChannelParameter(type=standard_artifacts.TransformGraph),
  }
  OUTPUTS = {
    'transformed_examples': ChannelParameter(type=standard_artifacts.Examples)
  }


class TransformWithGraph(base_component.BaseComponent):
  """Custom TFX Slack Component.

  This custom component serves as a bridge between TFX pipeline and human model
  reviewers to enable review-and-push workflow in model development cycle. It
  utilizes Slack API to send message to user-defined Slack channel with model
  URI info and wait for go / no-go decision from the same Slack channel:
    * To approve the model, a user need to reply the thread sent out by the bot
      started by SlackComponent with 'lgtm' or 'approve'.
    * To reject the model, a user need to reply the thread sent out by the bot
      started by SlackComponent with 'decline' or 'reject'.

  If the model is approved, an artifact will be created in ML metadata. It will
  be materialized as a file named 'BLESSED' in the directory specified by the
  URI of 'slack_blessing' artifact.
  If the model is rejected, an artifact will be created in ML metadata. It will
  be materialized as a file named 'NOT_BLESSED' in the directory specified by
  the URI of 'slack_blessing' channel.
  If no message indicating approve or reject was is received within given within
  timeout_sec, component will error out. This ensures that model will not be
  pushed and the validation is still retry-able.

  The output artifact might contain the following custom properties:
    - blessed: integer value indicating whether the model is blessed
    - slack_decision_maker: the user id that made the decision.
    - slack_decision_message: the message of the decision
    - slack_decision_channel: the slack channel the decision is made on
    - slack_decision_thread: the slack thread the decision is made on
  """

  SPEC_CLASS = TransformWithGraphSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(Executor)

  def __init__(
    self,
    examples: types.Channel,
    schema: types.Channel,
    transform_graph: types.Channel,
    transformed_examples: Optional[types.Channel] = None,
    instance_name: Optional[Text] = None):
    """Construct a SlackComponent.

    Args:
      model: A Channel of 'ModelExportPath' type, usually produced by
        Trainer component.
      model_blessing: A Channel of 'ModelBlessingPath' type, usually produced by
        ModelValidator component.
      slack_token: A token used for setting up connection with Slack server.
      slack_channel_id: Slack channel id to communicate on.
      timeout_sec: Seconds to wait for response before default to reject.
      slack_blessing: Optional output channel of 'ModelBlessingPath' with result
        of blessing; will be created for you if not specified.
    """
    if not transformed_examples:
      example_artifact = standard_artifacts.Examples()
      example_artifact.split_names = artifact_utils.encode_split_names(artifact.DEFAULT_EXAMPLE_SPLITS)
      transformed_examples = types.Channel(
          type=standard_artifacts.Examples, artifacts=[example_artifact])
    spec = TransformWithGraphSpec(
      examples=examples,
      schema=schema,
      transform_graph=transform_graph,
      transformed_examples=transformed_examples
    )
    super(TransformWithGraph, self).__init__(spec=spec, instance_name=instance_name)
