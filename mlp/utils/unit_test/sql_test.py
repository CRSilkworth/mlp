"""Tests for pandas_window_example_gen component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import unittest
import tempfile

from mlp.utils.sql import rolling_query, top_values


class DirsTest(unittest.TestCase):
  def test_rolling_query(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      query_file_name = os.path.join(temp_dir, 'query.sql')
      with open(query_file_name, 'w') as query_file:
        query_file.write('SELECT\n')
        query_file.write('  {select_columns}\n')
        query_file.write('FROM fake_table')
      query = rolling_query(
        query_file_name,
        partition_field='part',
        order_field='ord',
        num_leading=4,
        field_names=['field_a', 'field_b']
      )
      self.assertEqual(len(query), 669)

  def test_top_values(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      query_file_name = os.path.join(temp_dir, 'query.sql')
      with open(query_file_name, 'w') as query_file:
        query_file.write('SELECT {select_columns}\n')
        query_file.write('FROM fake_table')
      query = top_values(
        query_file_name,
        order_field='ord',
        top_k=4,
        other_select=['field_1', 'field_2'],
        field_names=['field_a', 'field_b']
      )
      self.assertEqual(len(query), 418)


if __name__ == '__main__':
  unittest.main()
  # tf.test.main()
