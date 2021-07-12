"""Tests for pandas_window_example_gen component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import unittest
import tempfile

from mlp.utils.dirs import copy_dir, download_dir


class DirsTest(unittest.TestCase):

  def test_copy_dir(self):
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
          self._touch(os.path.join(artifact_path, 'file_1.txt'))
          self._touch(os.path.join(artifact_path, 'file_2.txt'))
        src_pipeline_uri = os.path.join(base_dir, '2020-08-01-09-05-00')
        dst_pipeline_uri = os.path.join(base_dir, '2020-08-01-09-05-01')
        next_pipeline_uri = os.path.join(base_dir, '2020-08-01-09-05-02')

        copy_dir(src_pipeline_uri, dst_pipeline_uri)
        copy_dir(dst_pipeline_uri, next_pipeline_uri)

  def test_download_dir(self):
    src_uri = 'gs://biwako-prd/tmp/model/1/eval_model_dir'
    with tempfile.TemporaryDirectory() as temp_dir:
      download_dir(src_uri, temp_dir)
      print(temp_dir)

  def _touch(self, path):
    with open(path, 'a'):
        os.utime(path, None)



if __name__ == '__main__':
  # unittest.main()
  tf.test.main()
