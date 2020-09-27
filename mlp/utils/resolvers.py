import tensorflow as tf
import datetime
import os
import glob
from typing import Tuple, Text, Optional


class _TrainingPipelinePath(object):
  """Helper class to store data about kubeflow pipeline directories."""

  def __init__(
    self,
    full_path: Text,
    start_index: int,
    end_index: int):
    """Create _TrainingPipelinePath object.
    Parameters
    ----------
    full_path:
    start_index:
    end_index:



    """
    self.full_path = full_path
    self.start_ts = self._maybe_convert_to_ts(full_path.split('/')[start_index])
    self.end_ts = self._maybe_convert_to_ts(full_path.split('/')[end_index])
    self.uri = '/'.join(full_path.rstrip('/').split('/')[:-5])

  def _maybe_convert_to_ts(
    self,
    str_datetime: datetime.datetime) -> int:
    """Try to convert datetime to timestamp. Or return if already timestamp."""
    try:
      return int(str_datetime)
    except ValueError:
      dt = datetime.datetime.strptime(str_datetime, '%Y-%m-%d-%H-%M-%S')
      ts = datetime.datetime.timestamp(dt)

    return int(ts)


def multi_pipeline_uri(
  base_full_dir: Text,
  base_incremental_dir: Text,
  mid_path: Optional[Text] = 'data/AlwaysPusher/pushed_model/*'
  ) -> Text:
  """Selects out the most recently completely pipeline, giving preference to the full pipeline completions.

  Parameters
  ----------
  base_full_dir: The directory of all the full e2e pipelines.
  base_incremental_dir: The directory of all the incremental pipelines.
  mid_path: The relative directory of the completed model (This is how it checks whether oa model is completed or not).

  Returns
  -------
  latest_pipeline_uri: The uri of the most recently successfully completed pipeline.

  """
  start_index = len(base_full_dir.rstrip('/').split('/'))
  end_index = start_index + len(mid_path.lstrip('/').split('/')) + 1

  # Pull in all the full completed e22 pipelines, create _TrainingPipelinePaths.
  full_pattern = os.path.join(base_full_dir, '*', mid_path, '*/')
  if full_pattern.startswith('gs://'):
    full_dirs = tf.io.gfile.glob(full_pattern)
  else:
    full_dirs = glob.glob(full_pattern)

  full_tpps = [
    _TrainingPipelinePath(f_dir, start_index, end_index) for f_dir in full_dirs
  ]

  # Get the last completed full
  last_full = None
  if full_tpps:
    last_full = max(full_tpps, key=lambda x: x.start_ts)

  # Pull in the completed incremental pipelines, create _TrainingPipelinePaths.
  incremental_pattern = os.path.join(base_incremental_dir, '*', mid_path, '*/')
  if incremental_pattern.startswith('gs://'):
    incremental_dirs = tf.io.gfile.glob(incremental_pattern)
  else:
    incremental_dirs = glob.glob(incremental_pattern)

  incremental_tpps = [
    _TrainingPipelinePath(f_dir, start_index, end_index) for f_dir in incremental_dirs
  ]

  # Get the last completed incremental
  last_incremental = None
  if incremental_tpps:
    last_incremental = max(incremental_tpps, key=lambda x: x.start_ts)

  # Require that there is exist some completed pipeline
  if not full_tpps and not incremental_tpps:
    raise ValueError(
      "No models matching {} or {}".format(full_pattern, incremental_pattern)
    )


  elif not full_tpps:
    return last_incremental.uri

  elif not incremental_tpps:
    return last_full.uri

  if last_incremental.start_ts <= last_full.end_ts and last_full.end_ts <= last_incremental.end_ts:
    return last_full.uri
  elif last_incremental.end_ts > last_full.end_ts:
    return last_incremental.uri

  return last_full.uri


def latest_artifact_path(
  pipeline_uri: Text,
  artifact_path: Optional[Text] = 'data/AlwaysPusher/pushed_model') -> Text:
  uri = os.path.join(pipeline_uri, artifact_path, '*')
  if uri.startswith('gs://'):
    uris = tf.io.gfile.glob(uri)
  else:
    uris = glob.glob(uri)

  max_num = None
  latest_uri = None
  for uri in uris:
    num = int(uri.rstrip('/').split('/')[-1])

    if max_num is None or max_num < num:
      max_num = num
      latest_uri = uri

  return latest_uri


def latest_incremental_artifacts(
  pipeline_uri: Text) -> Tuple[Text, Text, Text]:
  schema_uri = latest_artifact_path(pipeline_uri, artifact_path='serving/schema')
  transform_graph_uri = latest_artifact_path(pipeline_uri, artifact_path='serving/transform_graph')
  model_uri = latest_artifact_path(pipeline_uri, artifact_path='serving/model')

  return schema_uri, transform_graph_uri, model_uri
