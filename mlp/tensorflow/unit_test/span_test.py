"""Tests for spans."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_text as tf_text
from tensorflow.python.lib.io import file_io
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

    new_span_starts, new_span_ends = self._old_to_new_format(strings, span_starts, span_ends, vocab_file_path)

    self.assertAllEqual(new_span_starts, [[3], [1], [1]])
    self.assertAllEqual(new_span_ends, [[5], [2], [3]])

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

    new_span_starts, new_span_ends = self._old_to_new_format(strings, span_starts, span_ends, vocab_file_path)

    self.assertAllEqual(new_span_starts, [[3], [1], [1]])
    self.assertAllEqual(new_span_ends, [[5], [2], [3]])

  def test__chat_to_wordpiece_spans_overflow(self):
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

    new_span_starts, new_span_ends = self._old_to_new_format(strings, span_starts, span_ends, vocab_file_path, max_seq_length=3)

    self.assertAllEqual(new_span_starts, [[3], [1], [1]])
    self.assertAllEqual(new_span_ends, [[3], [2], [3]])

    new_span_starts, new_span_ends = self._old_to_new_format(strings, span_starts, span_ends, vocab_file_path, max_seq_length=4)

    self.assertAllEqual(new_span_starts, [[3], [1], [1]])
    self.assertAllEqual(new_span_ends, [[4], [2], [3]])

    new_span_starts, new_span_ends = self._old_to_new_format(strings, span_starts, span_ends, vocab_file_path, max_seq_length=5)

    self.assertAllEqual(new_span_starts, [[3], [1], [1]])
    self.assertAllEqual(new_span_ends, [[5], [2], [3]])

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

    new_span_starts, new_span_ends = self._old_to_new_format(strings, span_starts, span_ends, vocab_file_path, max_seq_length=3)

    self.assertAllEqual(new_span_starts, [[3, 0], [1, 0]])
    self.assertAllEqual(new_span_ends, [[3, 1], [2, 1]])

  def _old_to_new_format(self, strings, span_starts, span_ends, vocab_file_path, max_seq_length=512, begin_token=''):
      with file_io.FileIO(vocab_file_path, mode='r') as vocab_file:
        vocab = vocab_file.read().split()
      init = tf.lookup.KeyValueTensorInitializer(
        vocab,
        tf.range(tf.size(vocab, out_type=tf.int64), dtype=tf.int64),
        key_dtype=tf.string,
        value_dtype=tf.int64
      )
      vocab_table = tf.lookup.StaticVocabularyTable(
        init, 1, lookup_key_dtype=tf.string
      )

      # Define the tokenizers
      whitespace_tokenizer = tf_text.WhitespaceTokenizer()
      tokenizer = tf_text.WordpieceTokenizer(
        vocab_table,
        token_out_type=tf.string
      )
      ws_tokens, ws_start, ws_end = whitespace_tokenizer.tokenize_with_offsets(begin_token + strings)
      wp_tokens, wp_start, wp_end = tokenizer.tokenize_with_offsets(ws_tokens)

      # Cast all integers to int64.
      ws_start, ws_end = tf.cast(ws_start, tf.int64), tf.cast(ws_end, tf.int64)
      wp_start, wp_end = tf.cast(wp_start, tf.int64), tf.cast(wp_end, tf.int64)

      return char_to_wordpiece_spans(
        ws_tokens=ws_tokens,
        ws_start=ws_start,
        ws_end=ws_end,
        wp_tokens=wp_tokens,
        wp_start=wp_start,
        wp_end=wp_end,
        char_start=span_starts,
        char_end=span_ends,
        max_seq_length=max_seq_length)

if __name__ == '__main__':
  tf.test.main()
