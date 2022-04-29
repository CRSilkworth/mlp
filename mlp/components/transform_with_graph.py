
"""TFX Transform component definition."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from typing import Any, Dict, Optional, Text, Union, List, Set, Sequence, Mapping, Callable
from tensorflow_metadata.proto.v0 import schema_pb2

import absl

import tensorflow as tf
import apache_beam as beam
import tensorflow_data_validation as tfdv

from tfx import types
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec
from tfx.components.transform import executor
from tfx.orchestration import data_types
from tfx.types import standard_artifacts
from tfx.types.standard_component_specs import TransformSpec

from tfx.types.component_spec import ComponentSpec
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ExecutionParameter
from tfx.components.transform import stats_options_util
from tfx.components.transform import labels
from tfx.types import artifact
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.components.util import tfxio_utils
from tfx.components.util import value_utils

import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.beam import analyzer_cache
from tensorflow_transform.beam.tft_beam_io import transform_fn_io

EXAMPLES_KEY = 'examples'
# Key for schema in executor input_dict.
SCHEMA_KEY = 'schema'

# Key for temp path, for internal use only.
TEMP_PATH_KEY = 'temp_path'

# Key for transform graph in executor output_dict.
TRANSFORM_GRAPH_KEY = 'transform_graph'
TRANSFORM_OUTPUT_KEY = 'transform_output'
# Key for output model in executor output_dict.
TRANSFORMED_EXAMPLES_KEY = 'transformed_examples'

RAW_EXAMPLE_KEY = 'raw_example'

# Schema to use if the input data should be decoded as raw example.
_RAW_EXAMPLE_SCHEMA = schema_utils.schema_from_feature_spec(
    {RAW_EXAMPLE_KEY: tf.io.FixedLenFeature([], tf.string)})

# TODO(b/123519698): Simplify the code by removing the key structure.
_TRANSFORM_INTERNAL_FEATURE_FOR_KEY = '__TFT_PASS_KEY__'

# Default file name prefix for transformed_examples.
_DEFAULT_TRANSFORMED_EXAMPLES_PREFIX = 'transformed_examples'

# Temporary path inside transform_output used for tft.beam
# TODO(b/125451545): Provide a safe temp path from base executor instead.
_TEMP_DIR_IN_TRANSFORM_OUTPUT = '.temp_path'

_TRANSFORM_COMPONENT_DESCRIPTOR = 'Transform'
_TELEMETRY_DESCRIPTORS = [_TRANSFORM_COMPONENT_DESCRIPTOR]

# TODO(b/37788560): Increase this max, based on results of experimentation with
# many non-packable analyzers on our benchmarks.
_MAX_ESTIMATED_STAGES_COUNT = 20000


def _InvokeStatsOptionsUpdaterFn(
    stats_options_updater_fn: Callable[
        [stats_options_util.StatsType, tfdv.StatsOptions], tfdv.StatsOptions],
    stats_type: stats_options_util.StatsType,
    schema: Optional[schema_pb2.Schema] = None,
    asset_map: Optional[Dict[Text, Text]] = None,
    transform_output_path: Optional[Text] = None) -> tfdv.StatsOptions:
  """Invokes the provided stats_options_updater_fn.

  Args:
    stats_options_updater_fn: The function to call.
    stats_type: The stats_type use in the function call.
    schema: The input schema to use in the function call.
    asset_map: A dictionary containing key to filename mappings.
    transform_output_path: The path to the transform output.

  Returns:
    The updated tfdv.StatsOptions.
  """
  options = {}
  if schema is not None:
    schema_copy = schema_pb2.Schema()
    schema_copy.CopyFrom(schema)
    options['schema'] = schema_copy
  if asset_map is not None:
    asset_path = os.path.join(transform_output_path, 'transform_fn',
                              tf.saved_model.ASSETS_DIRECTORY)
    vocab_paths = {k: os.path.join(asset_path, v) for k, v in asset_map.items()}
    options['vocab_paths'] = vocab_paths
  return stats_options_updater_fn(stats_type, tfdv.StatsOptions(**options))


class TransformWithGraphSpec(ComponentSpec):
  """Transform component spec."""

  PARAMETERS = {
    'custom_config': ExecutionParameter(type=(str, Text), optional=True),
  }
  INPUTS = {
    'examples': ChannelParameter(type=standard_artifacts.Examples),
    'schema': ChannelParameter(type=standard_artifacts.Schema),
    'transform_graph': ChannelParameter(type=standard_artifacts.TransformGraph),
  }
  OUTPUTS = {
    'transform_output': ChannelParameter(type=standard_artifacts.TransformGraph),
    'transformed_examples': ChannelParameter(type=standard_artifacts.Examples, optional=True),
  }
  # TODO(b/139281215): these input / output names have recently been renamed.
  # These compatibility aliases are temporarily provided for backwards
  # compatibility.
  _INPUT_COMPATIBILITY_ALIASES = {
      'input_data': 'examples',
  }
  _OUTPUT_COMPATIBILITY_ALIASES = {}


class TransformWithGraphExecutor(executor.Executor):
  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """TensorFlow Transform executor entrypoint.

    This implements BaseExecutor.Do() and is invoked by orchestration systems.
    This is not inteded for manual usage or further customization. Please use
    the Transform() function which takes an input format with no artifact
    dependency.

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - input_data: A list of type `standard_artifacts.Examples` which
          should contain two splits 'train' and 'eval'.
        - schema: A list of type `standard_artifacts.Schema` which should
          contain a single schema artifact.
      output_dict: Output dict from key to a list of artifacts, including:
        - transform_output: Output of 'tf.Transform', which includes an exported
          Tensorflow graph suitable for both training and serving;
        - transformed_examples: Materialized transformed examples, which
          includes both 'train' and 'eval' splits.
      exec_properties: A dict of execution properties, including either one of:
        - module_file: The file path to a python module file, from which the
          'preprocessing_fn' function will be loaded.
        - preprocessing_fn: The module path to a python function that
          implements 'preprocessing_fn'.

    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    train_data_uri = artifact_utils.get_split_uri(input_dict[EXAMPLES_KEY],
                                                  'train')
    eval_data_uri = artifact_utils.get_split_uri(input_dict[EXAMPLES_KEY],
                                                 'eval')
    payload_format, data_view_uri = (
        tfxio_utils.resolve_payload_format_and_data_view_uri(
            input_dict[EXAMPLES_KEY]))
    schema_file = io_utils.get_only_uri_in_dir(
        artifact_utils.get_single_uri(input_dict[SCHEMA_KEY]))

    transform_graph_uri = artifact_utils.get_single_uri(
        input_dict[TRANSFORM_GRAPH_KEY])
    transform_output = artifact_utils.get_single_uri(
        output_dict[TRANSFORM_OUTPUT_KEY])

    temp_path = os.path.join(transform_output, _TEMP_DIR_IN_TRANSFORM_OUTPUT)
    absl.logging.debug('Using temp path %s for tft.beam', temp_path)

    materialize_output_paths = []
    if output_dict.get(TRANSFORMED_EXAMPLES_KEY) is not None:
      transformed_example_artifact = artifact_utils.get_single_instance(
          output_dict[TRANSFORMED_EXAMPLES_KEY])
      # TODO(b/161490287): move the split_names setting to executor for all
      # components.
      transformed_example_artifact.split_names = (
          artifact_utils.encode_split_names(artifact.DEFAULT_EXAMPLE_SPLITS))
      transformed_train_output = artifact_utils.get_split_uri(
          output_dict[TRANSFORMED_EXAMPLES_KEY], 'train')
      transformed_eval_output = artifact_utils.get_split_uri(
          output_dict[TRANSFORMED_EXAMPLES_KEY], 'eval')
      materialize_output_paths = [
          os.path.join(transformed_train_output,
                       _DEFAULT_TRANSFORMED_EXAMPLES_PREFIX),
          os.path.join(transformed_eval_output,
                       _DEFAULT_TRANSFORMED_EXAMPLES_PREFIX)
      ]

    def _GetCachePath(label, params_dict):
      if label not in params_dict:
        return None
      else:
        return artifact_utils.get_single_uri(params_dict[label])

    label_inputs = {
        'transform_graph_uri': transform_graph_uri,
        labels.COMPUTE_STATISTICS_LABEL:
            False,
        labels.SCHEMA_PATH_LABEL:
            schema_file,
        labels.EXAMPLES_DATA_FORMAT_LABEL:
            payload_format,
        labels.DATA_VIEW_LABEL:
            data_view_uri,
        labels.ANALYZE_DATA_PATHS_LABEL:
            io_utils.all_files_pattern(train_data_uri),
        labels.ANALYZE_PATHS_FILE_FORMATS_LABEL:
            labels.FORMAT_TFRECORD,
        labels.TRANSFORM_DATA_PATHS_LABEL: [
            io_utils.all_files_pattern(train_data_uri),
            io_utils.all_files_pattern(eval_data_uri)
        ],
        labels.TRANSFORM_PATHS_FILE_FORMATS_LABEL: [
            labels.FORMAT_TFRECORD, labels.FORMAT_TFRECORD
        ],
        labels.CUSTOM_CONFIG:
            exec_properties.get('custom_config', None),
    }
    cache_input = _GetCachePath('cache_input_path', input_dict)
    if cache_input is not None:
      label_inputs[labels.CACHE_INPUT_PATH_LABEL] = cache_input

    label_outputs = {
        labels.TRANSFORM_METADATA_OUTPUT_PATH_LABEL: transform_output,
        labels.TRANSFORM_MATERIALIZE_OUTPUT_PATHS_LABEL:
            materialize_output_paths,
        labels.TEMP_OUTPUT_LABEL: str(temp_path),
    }
    cache_output = _GetCachePath('cache_output_path', output_dict)
    if cache_output is not None:
      label_outputs[labels.CACHE_OUTPUT_PATH_LABEL] = cache_output
    status_file = 'status_file'  # Unused
    self.Transform(label_inputs, label_outputs, status_file)
    absl.logging.debug('Cleaning up temp path %s on executor success',
                       temp_path)
    io_utils.delete_dir(temp_path)

  def Transform(self, inputs: Mapping[Text, Any], outputs: Mapping[Text, Any],
                status_file: Text) -> None:
    """Executes on request.

    This is the implementation part of transform executor. This is intended for
    using or extending the executor without artifact dependency.

    Args:
      inputs: A dictionary of labelled input values, including:
        - labels.COMPUTE_STATISTICS_LABEL: Whether compute statistics.
        - labels.SCHEMA_PATH_LABEL: Path to schema file.
        - labels.EXAMPLES_DATA_FORMAT_LABEL: Example data format, one of the
            enums from example_gen_pb2.PayloadFormat.
        - labels.ANALYZE_DATA_PATHS_LABEL: Paths or path patterns to analyze
          data.
        - labels.ANALYZE_PATHS_FILE_FORMATS_LABEL: File formats of paths to
          analyze data.
        - labels.TRANSFORM_DATA_PATHS_LABEL: Paths or path patterns to transform
          data.
        - labels.TRANSFORM_PATHS_FILE_FORMATS_LABEL: File formats of paths to
          transform data.
        - labels.CUSTOM_CONFIG: Dictionary of additional parameters for
          preprocessing_fn, optional.
        - labels.DATA_VIEW_LABEL: DataView to be used to read the Example,
          optional
      outputs: A dictionary of labelled output values, including:
        - labels.PER_SET_STATS_OUTPUT_PATHS_LABEL: Paths to statistics output,
          optional.
        - labels.TRANSFORM_METADATA_OUTPUT_PATH_LABEL: A path to
          TFTransformOutput output.
        - labels.TRANSFORM_MATERIALIZE_OUTPUT_PATHS_LABEL: Paths to transform
          materialization.
        - labels.TEMP_OUTPUT_LABEL: A path to temporary directory.
      status_file: Where the status should be written (not yet implemented)
    """
    del status_file  # unused

    absl.logging.debug(
        'Inputs to executor.Transform function: {}'.format(inputs))
    absl.logging.debug(
        'Outputs to executor.Transform function: {}'.format(outputs))

    transform_graph_uri = value_utils.GetSoleValue(
        inputs, 'transform_graph_uri')

    compute_statistics = value_utils.GetSoleValue(
        inputs, labels.COMPUTE_STATISTICS_LABEL)
    transform_output_path = value_utils.GetSoleValue(
        outputs, labels.TRANSFORM_METADATA_OUTPUT_PATH_LABEL)
    raw_examples_data_format = value_utils.GetSoleValue(
        inputs, labels.EXAMPLES_DATA_FORMAT_LABEL)
    stats_options_updater_fn = self._GetStatsOptionsUpdaterFn(inputs)
    schema = value_utils.GetSoleValue(inputs, labels.SCHEMA_PATH_LABEL)
    input_dataset_metadata = self._ReadMetadata(raw_examples_data_format,
                                                schema)
    materialize_output_paths = value_utils.GetValues(
        outputs, labels.TRANSFORM_MATERIALIZE_OUTPUT_PATHS_LABEL)
    # preprocessing_fn = self._GetPreprocessingFn(inputs, outputs)
    per_set_stats_output_paths = value_utils.GetValues(
        outputs, labels.PER_SET_STATS_OUTPUT_PATHS_LABEL)
    analyze_data_paths = value_utils.GetValues(inputs,
                                               labels.ANALYZE_DATA_PATHS_LABEL)
    analyze_paths_file_formats = value_utils.GetValues(
        inputs, labels.ANALYZE_PATHS_FILE_FORMATS_LABEL)
    transform_data_paths = value_utils.GetValues(
        inputs, labels.TRANSFORM_DATA_PATHS_LABEL)
    transform_paths_file_formats = value_utils.GetValues(
        inputs, labels.TRANSFORM_PATHS_FILE_FORMATS_LABEL)
    # input_cache_dir = value_utils.GetSoleValue(
    #     inputs, labels.CACHE_INPUT_PATH_LABEL, strict=False)
    output_cache_dir = value_utils.GetSoleValue(
        outputs, labels.CACHE_OUTPUT_PATH_LABEL, strict=False)
    per_set_stats_output_paths = value_utils.GetValues(
        outputs, labels.PER_SET_STATS_OUTPUT_PATHS_LABEL)
    temp_path = value_utils.GetSoleValue(outputs, labels.TEMP_OUTPUT_LABEL)
    data_view_uri = value_utils.GetSoleValue(
        inputs, labels.DATA_VIEW_LABEL, strict=False)

    absl.logging.debug('Analyze data patterns: %s',
                       list(enumerate(analyze_data_paths)))
    absl.logging.debug('Transform data patterns: %s',
                       list(enumerate(transform_data_paths)))
    absl.logging.debug('Transform materialization output paths: %s',
                       list(enumerate(materialize_output_paths)))
    absl.logging.debug('Transform output path: %s', transform_output_path)

    if len(analyze_data_paths) != len(analyze_paths_file_formats):
      raise ValueError(
          'size of analyze_data_paths and '
          'analyze_paths_file_formats do not match: {} v.s {}'.format(
              len(analyze_data_paths), len(analyze_paths_file_formats)))
    if len(transform_data_paths) != len(transform_paths_file_formats):
      raise ValueError(
          'size of transform_data_paths and '
          'transform_paths_file_formats do not match: {} v.s {}'.format(
              len(transform_data_paths), len(transform_paths_file_formats)))

    can_process_analysis_jointly = not bool(output_cache_dir)
    analyze_data_list = self._MakeDatasetList(analyze_data_paths,
                                              analyze_paths_file_formats,
                                              raw_examples_data_format,
                                              data_view_uri,
                                              can_process_analysis_jointly)
    if not analyze_data_list:
      raise ValueError('Analyze data list must not be empty.')

    can_process_transform_jointly = not bool(per_set_stats_output_paths or
                                             materialize_output_paths)
    transform_data_list = self._MakeDatasetList(transform_data_paths,
                                                transform_paths_file_formats,
                                                raw_examples_data_format,
                                                data_view_uri,
                                                can_process_transform_jointly,
                                                per_set_stats_output_paths,
                                                materialize_output_paths)

    all_datasets = analyze_data_list + transform_data_list
    for d in all_datasets:
      d.tfxio = self._CreateTFXIO(d, input_dataset_metadata.schema)
    self._AssertSameTFXIOSchema(all_datasets)
    # typespecs = all_datasets[0].tfxio.TensorAdapter().OriginalTypeSpecs()

    # Inspecting the preprocessing_fn even if we know we need a full pass in
    # order to fail faster if it fails.
    # analyze_input_columns = tft.get_analyze_input_columns(
    #     preprocessing_fn, typespecs)
    #
    # if not compute_statistics and not materialize_output_paths:
    #   if analyze_input_columns:
    #     absl.logging.warning(
    #         'Not using the in-place Transform because the following features '
    #         'require analyzing: {}'.format(
    #             tuple(c for c in analyze_input_columns)))
    #   else:
    #     absl.logging.warning(
    #         'Using the in-place Transform since compute_statistics=False, '
    #         'it does not materialize transformed data, and the configured '
    #         'preprocessing_fn appears to not require analyzing the data.')
    #     self._RunInPlaceImpl(preprocessing_fn, input_dataset_metadata,
    #                          typespecs, transform_output_path)
    #     # TODO(b/122478841): Writes status to status file.
    #     return

    materialization_format = (
        transform_paths_file_formats[-1] if materialize_output_paths else None)
    self._RunBeamImpl(analyze_data_list, transform_data_list,
                      transform_graph_uri, stats_options_updater_fn,
                      input_dataset_metadata,
                      transform_output_path, raw_examples_data_format,
                      temp_path,
                      compute_statistics, per_set_stats_output_paths,
                      materialization_format, len(analyze_data_paths))
  # TODO(b/122478841): Writes status to status file.

  def _RunBeamImpl(self, analyze_data_list: List[executor._Dataset],
                   transform_data_list: List[executor._Dataset], transform_graph_uri: Text,
                   stats_options_updater_fn: Callable[
                       [stats_options_util.StatsType, tfdv.StatsOptions],
                       tfdv.StatsOptions],
                   input_dataset_metadata: dataset_metadata.DatasetMetadata,
                   transform_output_path: Text, raw_examples_data_format: int,
                   temp_path: Text, compute_statistics: bool,
                   per_set_stats_output_paths: Sequence[Text],
                   materialization_format: Optional[Text],
                   analyze_paths_count: int) -> executor._Status:
    """Perform data preprocessing with TFT.

    Args:
      analyze_data_list: List of datasets for analysis.
      transform_data_list: List of datasets for transform.
      preprocessing_fn: The tf.Transform preprocessing_fn.
      input_dataset_metadata: A DatasetMetadata object for the input data.
      transform_output_path: An absolute path to write the output to.
      raw_examples_data_format: The data format of the raw examples. One of the
        enums from example_gen_pb2.PayloadFormat.
      temp_path: A path to a temporary dir.
      compute_statistics: A bool indicating whether or not compute statistics.
      per_set_stats_output_paths: Paths to per-set statistics output. If empty,
        per-set statistics is not produced.
      materialization_format: A string describing the format of the materialized
        data or None if materialization is not enabled.
      analyze_paths_count: An integer, the number of paths that should be used
        for analysis.

    Returns:
      Status of the execution.
    """
    self._AssertSameTFXIOSchema(analyze_data_list)
    unprojected_typespecs = (
        analyze_data_list[0].tfxio.TensorAdapter().OriginalTypeSpecs())

    tf_transform_output = tft.TFTransformOutput(transform_graph_uri)

    analyze_input_columns = tft.get_analyze_input_columns(
        tf_transform_output.transform_raw_features, unprojected_typespecs)
    transform_input_columns = tft.get_transform_input_columns(
        tf_transform_output.transform_raw_features, unprojected_typespecs)
    # Use the same dataset (same columns) for AnalyzeDataset and computing
    # pre-transform stats so that the data will only be read once for these
    # two operations.
    if compute_statistics:
      analyze_input_columns = list(
          set(list(analyze_input_columns) + list(transform_input_columns)))

    for d in analyze_data_list:
      d.tfxio = d.tfxio.Project(analyze_input_columns)

    self._AssertSameTFXIOSchema(analyze_data_list)
    analyze_data_tensor_adapter_config = (
        analyze_data_list[0].tfxio.TensorAdapterConfig())

    for d in transform_data_list:
      d.tfxio = d.tfxio.Project(transform_input_columns)

    desired_batch_size = self._GetDesiredBatchSize(raw_examples_data_format)

    with self._CreatePipeline(transform_output_path) as pipeline:
      with tft_beam.Context(
        temp_dir=temp_path,
        desired_batch_size=desired_batch_size,
        passthrough_keys=self._GetTFXIOPassthroughKeys(),
        use_deep_copy_optimization=True
      ):
        if compute_statistics or materialization_format is not None:
          transform_fn = (
              pipeline | transform_fn_io.ReadTransformFn(transform_graph_uri))

          # Do not compute pre-transform stats if the input format is raw proto,
          # as StatsGen would treat any input as tf.Example. Note that
          # tf.SequenceExamples are wire-format compatible with tf.Examples.
          if (compute_statistics and
              not self._IsDataFormatProto(raw_examples_data_format)):
            # Aggregated feature stats before transformation.
            pre_transform_feature_stats_path = os.path.join(
                transform_output_path,
                tft.TFTransformOutput.PRE_TRANSFORM_FEATURE_STATS_PATH)

            if self._IsDataFormatSequenceExample(raw_examples_data_format):
              schema_proto = None
            else:
              schema_proto = executor._GetSchemaProto(input_dataset_metadata)

            if self._IsDataFormatSequenceExample(raw_examples_data_format):
              def _ExtractRawExampleBatches(record_batch):
                return record_batch.column(
                    record_batch.schema.get_field_index(
                        RAW_EXAMPLE_KEY)).flatten().to_pylist()
              # Make use of the fact that tf.SequenceExample is wire-format
              # compatible with tf.Example
              stats_input = []
              for dataset in analyze_data_list:
                infix = 'AnalysisIndex{}'.format(dataset.index)
                stats_input.append(
                    dataset.standardized
                    | 'ExtractRawExampleBatches[{}]'.format(infix) >> beam.Map(
                        _ExtractRawExampleBatches)
                    | 'DecodeSequenceExamplesAsExamplesIntoRecordBatches[{}]'
                    .format(infix) >> beam.ParDo(
                        self._ToArrowRecordBatchesFn(schema_proto)))
            else:
              stats_input = [
                  dataset.standardized for dataset in analyze_data_list]

            pre_transform_stats_options = _InvokeStatsOptionsUpdaterFn(
                stats_options_updater_fn,
                stats_options_util.StatsType.PRE_TRANSFORM, schema_proto)
            (stats_input
             | 'FlattenAnalysisDatasets' >> beam.Flatten(pipeline=pipeline)
             | 'GenerateStats[FlattenedAnalysisDataset]' >> self._GenerateStats(
                 pre_transform_feature_stats_path,
                 schema_proto,
                 stats_options=pre_transform_stats_options))

          # transform_data_list is a superset of analyze_data_list, we pay the
          # cost to read the same dataset (analyze_data_list) again here to
          # prevent certain beam runner from doing large temp materialization.
          for dataset in transform_data_list:
            infix = 'TransformIndex{}'.format(dataset.index)
            dataset.standardized = (
                pipeline | 'TFXIOReadAndDecode[{}]'.format(infix) >>
                dataset.tfxio.BeamSource(desired_batch_size))
            (dataset.transformed, metadata) = (
                ((dataset.standardized, dataset.tfxio.TensorAdapterConfig()),
                 transform_fn)
                | 'Transform[{}]'.format(infix) >> tft_beam.TransformDataset())

            dataset.transformed_and_serialized = (
                dataset.transformed
                | 'EncodeAndSerialize[{}]'.format(infix)
                >> beam.ParDo(self._EncodeAsSerializedExamples(),
                              executor._GetSchemaProto(metadata)))

          if compute_statistics:
            # Aggregated feature stats after transformation.
            _, metadata = transform_fn

            # TODO(b/70392441): Retain tf.Metadata (e.g., IntDomain) in
            # schema. Currently input dataset schema only contains dtypes,
            # and other metadata is dropped due to roundtrip to tensors.
            transformed_schema_proto = executor._GetSchemaProto(metadata)

            for dataset in transform_data_list:
              infix = 'TransformIndex{}'.format(dataset.index)
              dataset.transformed_and_standardized = (
                  dataset.transformed_and_serialized
                  | 'FromTransformedToArrowRecordBatches[{}]'
                  .format(infix)
                  >> self._ToArrowRecordBatches(
                      schema=transformed_schema_proto))

            post_transform_feature_stats_path = os.path.join(
                transform_output_path,
                tft.TFTransformOutput.POST_TRANSFORM_FEATURE_STATS_PATH)

            post_transform_stats_options = _InvokeStatsOptionsUpdaterFn(
                stats_options_updater_fn,
                stats_options_util.StatsType.POST_TRANSFORM,
                transformed_schema_proto, metadata.asset_map,
                transform_output_path)
            ([dataset.transformed_and_standardized
              for dataset in transform_data_list]
             | 'FlattenTransformedDatasets' >> beam.Flatten()
             | 'GenerateStats[FlattenedTransformedDatasets]' >>
             self._GenerateStats(
                 post_transform_feature_stats_path,
                 transformed_schema_proto,
                 stats_options=post_transform_stats_options))

            if per_set_stats_output_paths:
              # TODO(b/130885503): Remove duplicate stats gen compute that is
              # done both on a flattened view of the data, and on each span
              # below.
              for dataset in transform_data_list:
                infix = 'TransformIndex{}'.format(dataset.index)
                (dataset.transformed_and_standardized
                 | 'GenerateStats[{}]'.format(infix) >> self._GenerateStats(
                     dataset.stats_output_path,
                     transformed_schema_proto,
                     stats_options=post_transform_stats_options))

          if materialization_format is not None:
            for dataset in transform_data_list:
              infix = 'TransformIndex{}'.format(dataset.index)
              (dataset.transformed_and_serialized
               | 'Materialize[{}]'.format(infix) >> self._WriteExamples(
                   materialization_format,
                   dataset.materialize_output_path))

    return executor._Status.OK()


class TransformWithGraph(base_component.BaseComponent):
  """A TFX component to transform the input examples.

  The Transform component wraps TensorFlow Transform (tf.Transform) to
  preprocess data in a TFX pipeline. This component will load the
  preprocessing_fn from input module file, preprocess both 'train' and 'eval'
  splits of input examples, generate the `tf.Transform` output, and save both
  transform function and transformed examples to orchestrator desired locations.

  ## Providing a preprocessing function
  The TFX executor will use the estimator provided in the `module_file` file
  to train the model.  The Transform executor will look specifically for the
  `preprocessing_fn()` function within that file.

  An example of `preprocessing_fn()` can be found in the [user-supplied
  code]((https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_utils.py))
  of the TFX Chicago Taxi pipeline example.

  ## Example
  ```
  # Performs transformations and feature engineering in training and serving.
  transform = Transform(
      examples=example_gen.outputs['examples'],
      schema=infer_schema.outputs['schema'],
      module_file=module_file)
  ```

  Please see https://www.tensorflow.org/tfx/transform for more details.
  """

  SPEC_CLASS = TransformWithGraphSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(TransformWithGraphExecutor)

  def __init__(
      self,
      examples: types.Channel = None,
      schema: types.Channel = None,
      transform_graph: types.Channel = None,
      transform_output: Optional[types.Channel] = None,
      transformed_examples: Optional[types.Channel] = None,
      materialize: bool = True,
      custom_config: Optional[Dict[Text, Any]] = None):
    """Construct a Transform component.

    Args:
      examples: A Channel of type `standard_artifacts.Examples` (required).
        This should contain the two splits 'train' and 'eval'.
      schema: A Channel of type `standard_artifacts.Schema`. This should
        contain a single schema artifact.
      transform_graph: Input for 'TransformPath' channel to get the
        'tf.Transform', which includes an exported Tensorflow graph suitable for
        both training and serving;
      transformed_examples: Optional output 'ExamplesPath' channel for
        materialized transformed examples, which includes both 'train' and
        'eval' splits.
      materialize: If True, write transformed examples as an output. If False,
        `transformed_examples` must not be provided.
      custom_config: A dict which contains additional parameters that will be
        passed to preprocessing_fn.

    Raises:
      ValueError: When both or neither of 'module_file' and 'preprocessing_fn'
        is supplied.
    """
    if materialize and transformed_examples is None:
      transformed_examples = types.Channel(
          type=standard_artifacts.Examples,
          # matching_channel_name='examples')
          )
    elif not materialize and transformed_examples is not None:
      raise ValueError(
          'Must not specify transformed_examples when materialize is False.')

    transform_output = transform_output or types.Channel(
        type=standard_artifacts.TransformGraph)
    spec = TransformWithGraphSpec(
        examples=examples,
        schema=schema,
        transform_graph=transform_graph,
        transform_output=transform_output,
        transformed_examples=transformed_examples,
        custom_config=json.dumps(custom_config))
    super(TransformWithGraph, self).__init__(spec=spec)
