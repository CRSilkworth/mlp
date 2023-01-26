"""Tests for spans."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_text as tf_text
from tensorflow.python.lib.io import file_io
from mlp.tensorflow.spans import char_to_wordpiece_spans, wordpiece_to_char_spans
from mlp.layers.bert import BertTokenizer
import numpy as np


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

  def test_wordpiece_to_char_spans_whitespace(self):
    begin_token = ''
    vocab_file_path = 'gs://biwako-prd/bert/data/vocab.txt'
    span_start = [[2, 0, 4, 7], [87, 34, 0, 10]]  # 8
    span_end = [[5, 4, 9, 7], [212, 36, 10, 20]]  # 18
    # span_start = [[2, 0, 4, 7]]  # 8
    # span_end = [[5, 4, 9, 7]]
    strings_list = [
      b'The  arsonist has oddly \nshaped feet',
      """【室内犬および室内猫の同伴について】

      ペット同伴をご希望のお客様は、宿泊利用規約にご同意いただく必要があります。

      ・ペット同伴可能な部屋は201号室、203号室、301号室、303号室です。202号室、302号室は対象外となります。
      ・体重10キロ未満の室内犬、室内猫に限ります。
      ・共用部では抱っこかキャリーバッグに入れてください。
      ・客室と屋上では自由にしていただけます。
      ・ペット用アメニティはございません。

      ペット同伴宿泊利用規約
      必ず[こちら](http://moopon.jp/dogs_Bar&Hotel.Colors.Miyajima.pdf)をご確認ください。""",
    ]
    strings = tf.constant(strings_list)
    span_start = tf.constant(span_start)
    span_end = tf.constant(span_end)

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
    bert_tokenizer = BertTokenizer('gs://biwako-prd/bert/data')
    whitespace_tokenizer = bert_tokenizer.basic_tokenizer
    tokenizer = tf_text.WordpieceTokenizer(
      vocab_table,
      token_out_type=tf.string
    )
    ws_tokens, ws_start, ws_end = whitespace_tokenizer.tokenize_with_offsets(begin_token + strings)
    wp_tokens, wp_start, wp_end = tokenizer.tokenize_with_offsets(ws_tokens)

    char_start, char_end = wordpiece_to_char_spans(
      ws_start,
      ws_end,
      wp_start,
      wp_end,
      span_start,
      span_end,
    )

    for row_num in range(0, 2):
      print('+'*100)
      # print(row_num)
      # print(strings[row_num])
      # print(span_start[row_num], span_end[row_num])
      for start, end in zip(span_start[row_num], span_end[row_num]):
        wp_text = bert_tokenizer._merge_dims(wp_tokens, -2)[row_num][start: end]
        wp_text = ' '.join([char.decode() for char in wp_text.numpy()])
        print(start.numpy(), end.numpy(), wp_text)

      print('='*100)

      for start, end in zip(char_start[row_num], char_end[row_num]):
        needed_text = (
            tf.strings.substr(
                tf.constant(strings[row_num]),
                start,
                end - start,
            )
            .numpy()
        )
        print(start.numpy(), end.numpy(), needed_text.decode())
      print('+'*100)
if __name__ == '__main__':
  tf.test.main()
