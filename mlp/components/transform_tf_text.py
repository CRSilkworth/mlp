import apache_beam as beam
from tensorflow_transform.beam import common as tft_beam_common
from tfx.components.transform.executor import Executor as TransformExecutor
from tfx.components.transform.component import Transform
from tfx.dsl.components.base import executor_spec


class Executor(TransformExecutor):
  @staticmethod
  @beam.ptransform_fn
  @beam.typehints.with_input_types(beam.Pipeline)
  @beam.typehints.with_output_types(beam.pvalue.PDone)
  def _IncrementPipelineMetrics(pipeline: beam.Pipeline,
                                total_columns_count: int,
                                analyze_columns_count: int,
                                transform_columns_count: int,
                                analyze_paths_count: int):
    """A beam PTransform to increment counters of column usage."""

    def _MakeAndIncrementCounters(unused_element):
      """Increment column usage counters."""
      del unused_element
      beam.metrics.Metrics.counter(
          tft_beam_common.METRICS_NAMESPACE,
          'total_columns_count').inc(total_columns_count)
      beam.metrics.Metrics.counter(
          tft_beam_common.METRICS_NAMESPACE,
          'analyze_columns_count').inc(analyze_columns_count)
      beam.metrics.Metrics.counter(
          tft_beam_common.METRICS_NAMESPACE,
          'transform_columns_count').inc(transform_columns_count)
      beam.metrics.Metrics.counter(
          tft_beam_common.METRICS_NAMESPACE,
          'analyze_paths_count').inc(analyze_paths_count)
      return beam.pvalue.PDone(pipeline)

    return (
        pipeline
        | 'CreateSole' >> beam.Create([None])
        | 'Count' >> beam.Map(_MakeAndIncrementCounters))
  # def _IncrementPipelineMetrics(
  #     pipeline: beam.Pipeline, total_columns_count: int,
  #     analyze_columns_count: int, transform_columns_count: int,
  #     analyze_paths_count: int, analyzer_cache_enabled: bool,
  #     disable_statistics: bool, materialize: bool):
  #   """A beam PTransform to increment counters of column usage."""
  #   import tensorflow_text as _
  #
  #   def _MakeAndIncrementCounters(unused_element):
  #     """Increment column usage counters."""
  #     del unused_element
  #     beam.metrics.Metrics.counter(
  #         tft_beam_common.METRICS_NAMESPACE,
  #         'total_columns_count').inc(total_columns_count)
  #     beam.metrics.Metrics.counter(
  #         tft_beam_common.METRICS_NAMESPACE,
  #         'analyze_columns_count').inc(analyze_columns_count)
  #     beam.metrics.Metrics.counter(
  #         tft_beam_common.METRICS_NAMESPACE,
  #         'transform_columns_count').inc(transform_columns_count)
  #     beam.metrics.Metrics.counter(
  #         tft_beam_common.METRICS_NAMESPACE,
  #         'analyze_paths_count').inc(analyze_paths_count)
  #     beam.metrics.Metrics.counter(
  #         tft_beam_common.METRICS_NAMESPACE,
  #         'analyzer_cache_enabled').inc(int(analyzer_cache_enabled))
  #     beam.metrics.Metrics.counter(
  #         tft_beam_common.METRICS_NAMESPACE,
  #         'disable_statistics').inc(int(disable_statistics))
  #     beam.metrics.Metrics.counter(
  #         tft_beam_common.METRICS_NAMESPACE,
  #         'materialize').inc(int(materialize))
  #     return beam.pvalue.PDone(pipeline)
  #
  #   return (
  #       pipeline
  #       | 'CreateSole' >> beam.Create([None])
  #       | 'Count' >> beam.Map(_MakeAndIncrementCounters))


class TransformTFText(Transform):
  # EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(Executor)
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(Executor)
