"""Tests for pandas_window_example_gen component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import unittest
import numpy as np
from mlp.layers.dense import IndexedDense


class DenseTest(unittest.TestCase):
  def test_index_dense(self):
    input_dim = 25
    batch_size = 10
    num_indices = 4
    output_dim = 7
    max_indices = 1
    inp = tf.keras.Input(shape=(input_dim,))
    ind = tf.keras.Input(shape=(1,), dtype=tf.int64)

    out = IndexedDense(output_dim, num_indices, activation='relu', dtype=tf.float64)([inp, ind])
    model = tf.keras.Model([inp, ind], out)

    input = np.random.normal(size=(batch_size, input_dim))
    index = np.random.randint(num_indices, size=(batch_size,)).astype(int)
    output = model.predict([input, index])

    self.assertEqual(output.shape, (batch_size, output_dim))

  def test_index_dense_3d(self):
    input_dim = 19
    sequence_length = 5
    batch_size = 10
    num_indices = 4
    output_dim = 7
    max_indices = 1
    inp = tf.keras.Input(shape=(sequence_length, input_dim))
    ind = tf.keras.Input(shape=(1,), dtype=tf.int64)

    out = IndexedDense(output_dim, num_indices, activation='relu', dtype=tf.float64)([inp, ind])
    model = tf.keras.Model([inp, ind], out)

    input = np.random.normal(size=(batch_size, sequence_length, input_dim))
    index = np.random.randint(num_indices, size=(batch_size,)).astype(int)
    output = model.predict([input, index])

    self.assertEqual(output.shape, (batch_size, sequence_length, output_dim))

  def test_index_dense_3d_index(self):
    input_dim = 19
    sequence_length = 5
    batch_size = 10
    num_indices = 11
    max_indices = 4
    output_dim = 7
    batch_dims = 0
    inp = tf.keras.Input(shape=(sequence_length, input_dim))
    ind = tf.keras.Input(shape=(max_indices, 1,), dtype=tf.int64)

    out = IndexedDense(output_dim, num_indices, activation='relu', dtype=tf.float64, batch_dims=batch_dims)([inp, ind])
    model = tf.keras.Model([inp, ind], out)

    input = np.random.normal(size=(batch_size, sequence_length, input_dim))
    index = np.random.randint(num_indices, size=(batch_size, max_indices, 1)).astype(int)
    output = model.predict([input, index])

    self.assertEqual(output.shape, (batch_size, sequence_length, max_indices, output_dim))
if __name__ == '__main__':
  unittest.main()
  # tf.test.main()
