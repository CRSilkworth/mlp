import tensorflow as tf
import datetime
import os


class _TrainingPipelinePath(object):

  def __init__(self, full_path, start_index, end_index):
    self.full_path = full_path
    self.start_ts = self._maybe_convert_to_ts(full_path.split('/')[start_index])
    self.end_ts = self._maybe_convert_to_ts(full_path.split('/')[end_index])
    self.uri = '/'.join(full_path.split('/')[:-5])

  def _maybe_convert_to_ts(self, str_datetime):
    # print('-'*20)
    # print(self.full_path)
    # print(str_datetime)
    try:
      return int(str_datetime)
    except ValueError:
      dt = datetime.datetime.strptime(str_datetime, '%Y-%m-%d-%H-%M-%S')
      ts = datetime.datetime.timestamp(dt)
      return int(ts)


def multi_pipeline_uri(base_full_dir, base_incremental_dir, mid_path='data/AlwaysPusher/pushed_model/*'):
  start_index = len(base_full_dir.rstrip('/').split('/'))
  end_index = start_index + len(mid_path.lstrip('/').split('/')) + 1

  full_pattern = os.path.join(base_full_dir, '*', mid_path, '*/')
  full_dirs = tf.io.gfile.glob(full_pattern)

  full_tpps = [
    _TrainingPipelinePath(f_dir, start_index, end_index) for f_dir in full_dirs
  ]
  last_full = None
  if full_tpps:
    last_full = max(full_tpps, key=lambda x: x.start_ts)

  incremental_pattern = os.path.join(base_incremental_dir, '*', mid_path, '*/')
  incremental_dirs = tf.io.gfile.glob(incremental_pattern)

  incremental_tpps = [
    _TrainingPipelinePath(f_dir, start_index, end_index) for f_dir in incremental_dirs
  ]
  last_incremental = None
  if incremental_tpps:
    last_incremental = max(incremental_tpps, key=lambda x: x.start_ts)

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
