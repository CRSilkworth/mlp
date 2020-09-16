"""Tests for pandas_window_example_gen component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import unittest
import tempfile

from mlp.utils.resolvers import multi_pipeline_uri, latest_artifact_path


class ComponentTest(unittest.TestCase):
  def test_multi_pipeline_base_model(self):
    full_dirs = [
      '2020-08-01-09-03-15/data/AlwaysPusher/pushed_model/18/2020-08-01-09-06-45',
    ]

    incremental_dirs = [
      '2020-08-01-09-05-00/data/AlwaysPusher/pushed_model/544/2020-08-01-09-05-50',
      '2020-08-01-09-06-00/data/AlwaysPusher/pushed_model/544/2020-08-01-09-06-50',
      '2020-08-01-09-07-00/data/AlwaysPusher/pushed_model/544/2020-08-01-09-07-50',
    ]
    for i in range(len(full_dirs) + 1):
      for j in range(len(incremental_dirs) + 1):
        with tempfile.TemporaryDirectory() as temp_dir:

          base_full_dir = os.path.join(temp_dir, 'runs/tfx/intent_classifier-polyglot-bigquery_to_pusher')
          tf.io.gfile.makedirs(base_full_dir)
          full_uris = []
          for full_dir in full_dirs[:i]:
            full_path = os.path.join(base_full_dir, full_dir)
            tf.io.gfile.makedirs(os.path.join(full_path, 'temp'))
            full_uris.append(self._remove_last_dir(full_path))

          base_incremental_dir = os.path.join(temp_dir, 'runs/tfx/intent_classifier-polyglot-incremental')
          tf.io.gfile.makedirs(base_incremental_dir)
          incremental_uris = []
          for incremental_dir in incremental_dirs[:j]:
            incremental_path = os.path.join(base_incremental_dir, incremental_dir)
            tf.io.gfile.makedirs(os.path.join(incremental_path, 'temp'))
            incremental_uris.append(self._remove_last_dir(incremental_path))

          if i == 0 and j == 0:
            with self.assertRaises(ValueError):
              model_path = multi_pipeline_uri(base_full_dir, base_incremental_dir)
          else:
            model_path = multi_pipeline_uri(base_full_dir, base_incremental_dir)
            if i == 0:
              self.assertEqual(model_path, os.path.join(base_incremental_dir, incremental_uris[j - 1]))
            elif j == 3:
              self.assertEqual(model_path, os.path.join(base_incremental_dir, incremental_uris[j - 1]))
            else:
              self.assertEqual(model_path, os.path.join(base_full_dir, full_uris[i - 1]))

  def test_latest_artifact_path(self):
    artifact_dirs = [
      '2020-08-01-09-05-00/data/AlwaysPusher/pushed_model/40/',
      '2020-08-01-09-05-00/data/AlwaysPusher/pushed_model/300/',
      '2020-08-01-09-05-00/data/AlwaysPusher/pushed_model/55/'
    ]
    for i in range(len(artifact_dirs)):
      with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = os.path.join(temp_dir, 'runs/tfx/pipeline')
        for artifact_dir in artifact_dirs[:i + 1]:
          artifact_path = os.path.join(base_dir, artifact_dir)
          tf.io.gfile.makedirs(artifact_path)

        pipeline_uri = os.path.join(base_dir, '2020-08-01-09-05-00')
        uri = latest_artifact_path(pipeline_uri, 'data/AlwaysPusher/pushed_model')

        if i == 0:
          self.assertEqual(uri.split('/')[-1], '40')
        else:
          self.assertEqual(uri.split('/')[-1], '300')

  def _remove_last_dir(self, dir):
    return '/'.join(dir.rstrip('/').split('/')[:-5])


if __name__ == '__main__':
  unittest.main()
  # tf.test.main()
