"""Tests for spans."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from mlp.tensorflow.spans import char_to_wordpiece_spans

class SpanTest(tf.test.TestCase):

  def test_char_to_wordpiece_spans(self):
    vocab_file_path = 'gs://biwako-prd/bert/data/vocab.txt'
    span_starts = [[8], [3], [2]]
    span_ends = [[18], [6], [10]]
    strings_list = [
      'I had a little dog named   mustard.',
      'He ran\n away with his friend\t named ketchup. The traitor.',
      'I never saw him again.'
    ]
    strings = tf.constant(strings_list)
    span_starts = tf.constant(span_starts)
    span_ends = tf.constant(span_ends)

    new_span_starts, new_span_ends = char_to_wordpiece_spans(strings, span_starts, span_ends, vocab_file_path)

    self.assertAllEqual(new_span_starts, [[4], [2], [2]])
    self.assertAllEqual(new_span_ends, [[6], [3], [4]])

  def test_char_to_wordpiece_spans_whitespace(self):
    vocab_file_path = 'gs://biwako-prd/bert/data/vocab.txt'
    span_starts = [[7], [2], [1]]
    span_ends = [[19], [7], [9]]
    strings_list = [
      'I had a little dog named   mustard.',
      'He ran\n away with his friend\t named ketchup. The traitor.',
      'I never saw him again.'
    ]
    strings = tf.constant(strings_list)
    span_starts = tf.constant(span_starts)
    span_ends = tf.constant(span_ends)

    new_span_starts, new_span_ends = char_to_wordpiece_spans(strings, span_starts, span_ends, vocab_file_path)

    self.assertAllEqual(new_span_starts, [[4], [2], [2]])
    self.assertAllEqual(new_span_ends, [[6], [3], [4]])

  def test_char_to_wordpiece_spans_overflow(self):
    vocab_file_path = 'gs://biwako-prd/bert/data/vocab.txt'
    span_starts = [[8], [3], [2]]
    span_ends = [[18], [6], [10]]
    strings_list = [
      'I had a little dog named   mustard.',
      'He ran\n away with his friend\t named ketchup. The traitor.',
      'I never saw him again.'
    ]
    strings = tf.constant(strings_list)
    span_starts = tf.constant(span_starts)
    span_ends = tf.constant(span_ends)

    new_span_starts, new_span_ends = char_to_wordpiece_spans(strings, span_starts, span_ends, vocab_file_path, max_seq_length=3)

    self.assertAllEqual(new_span_starts, [[3], [2], [2]])
    self.assertAllEqual(new_span_ends, [[3], [3], [3]])

    new_span_starts, new_span_ends = char_to_wordpiece_spans(strings, span_starts, span_ends, vocab_file_path, max_seq_length=4)

    self.assertAllEqual(new_span_starts, [[4], [2], [2]])
    self.assertAllEqual(new_span_ends, [[4], [3], [4]])

    new_span_starts, new_span_ends = char_to_wordpiece_spans(strings, span_starts, span_ends, vocab_file_path, max_seq_length=5)

    self.assertAllEqual(new_span_starts, [[4], [2], [2]])
    self.assertAllEqual(new_span_ends, [[5], [3], [4]])

  def test_char_to_wordpiece_spans_two_dims(self):
    vocab_file_path = 'gs://biwako-prd/bert/data/vocab.txt'
    span_starts = [[8, 0], [3, 0]]
    span_ends = [[18, 1], [6, 2]]
    strings_list = [
      'I had a little dog named   mustard.',
      'He ran\n away with his friend\t named ketchup. The traitor.',
    ]
    strings = tf.constant(strings_list)
    span_starts = tf.constant(span_starts)
    span_ends = tf.constant(span_ends)

    new_span_starts, new_span_ends = char_to_wordpiece_spans(strings, span_starts, span_ends, vocab_file_path, max_seq_length=3)

    self.assertAllEqual(new_span_starts, [[3, 1], [2, 1]])
    self.assertAllEqual(new_span_ends, [[3, 2], [3, 2]])

if __name__ == '__main__':
  tf.test.main()
