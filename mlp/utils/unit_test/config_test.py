"""Tests for pandas_window_example_gen component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import unittest
import tempfile

from mlp.utils.resolvers import multi_pipeline_uri, latest_artifact_path
from mlp.utils.config import VarConfig

class ComponentTest(unittest.TestCase):
  def test_config_save_load(self):

      with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, 'config', 'config_vars.py')
        vc = VarConfig()

        vc.a = 'a'
        vc.add_vars(
          b=1,
          c=[1, 2, 3]
        )
        vc.add_vars(
          c=[2, 3, 4],
        )
        vc.d = {'1': 'a', '2': 'b'}

        self.assertEqual(vc.vars, {'a': 'a', 'b':1, 'c':[2,3,4], 'd': {'1': 'a', '2': 'b'}})
        vc.write(file_path)

        del vc

        vc = VarConfig(file_path)

        self.assertEqual(vc.vars, {'a': 'a', 'b':1, 'c':[2,3,4], 'd': {'1': 'a', '2': 'b'}})


if __name__ == '__main__':
  unittest.main()
  # tf.test.main()
