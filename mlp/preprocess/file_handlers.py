from typing import Optional, Text, List
import tensorflow as tf


def _gzip_reader_fn(filenames: List[Text]) -> tf.data.TFRecordDataset:
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(
      filenames,
      compression_type='GZIP')
