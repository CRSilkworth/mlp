import tensorflow as tf
import tensorflow_text as tf_text
import os

from typing import Optional, List, Text, Dict, Any
from tensorflow.python.lib.io import file_io
from official.nlp import bert_modeling as modeling
from tensorflow_text.python.ops.bert_tokenizer import BasicTokenizer


class BertTokenizer(tf.keras.layers.Layer):
  def __init__(
    self,
    bert_dir: Text,
    max_seq_length: Optional[int] = 128,
    token_out_type: Optional[tf.DType] = tf.int64,
    basic_tokenizer_class=BasicTokenizer,
    **kwargs

    ):
    super(BertTokenizer, self).__init__(**kwargs)

    with tf.init_scope():
      self.vocab_file_path = os.path.join(bert_dir, 'vocab.txt')
      with file_io.FileIO(self.vocab_file_path, mode='r') as vocab_file:
        self.vocab = vocab_file.read().split()

      init = tf.lookup.KeyValueTensorInitializer(
        self.vocab,
        tf.range(tf.size(self.vocab, out_type=tf.int64), dtype=tf.int64),
        key_dtype=tf.string,
        value_dtype=tf.int64
      )

      self.vocab_table = tf.lookup.StaticVocabularyTable(
        init, 1, lookup_key_dtype=tf.string
      )
      self.basic_tokenizer = basic_tokenizer_class()
      self.tokenizer = tf_text.WordpieceTokenizer(
        self.vocab_table,
        token_out_type=token_out_type
      )

    self.max_seq_length = max_seq_length
    self.token_out_type = token_out_type

  def call(self, strings: tf.Tensor, training=False, begin_token='[CLS] ', **kwargs) -> tf.Tensor:
    """Convert a tensor of strings into a tensor of token ids.

    Uses the WordpieceTokenizer to convert a tensor of shape N to N+1. The added
    dimension is the 'sentence' dimension when the string gets converted into
    split up tokens.
    e.g.
    ["What time is it.", "it's 3 o'clock"] ->
    [
      [2, 3, 4, 5, 0, 0],
      [3, 5, 7, 1, 8, 0]
    ]

    Parameters
    ----------
    strings: A tensor of strings to be tokenized.

    Returns
    -------
    The tensor of tokenized strings.

    """
    # input_shape = strings.shape
    # output_shape = input_shape.concatenate(
    #   tf.TensorShape([self.max_seq_length]))
    tokens_dict = self.tokens_and_spans(strings, begin_token)

    # Collapse the ragged tensor dimension by one convert to a regular tensor.
    tokens = self.normalize_shape(tokens_dict['wp_tokens'])

    return tokens

  @tf.function
  def tokens_and_spans(self, strings, begin_token='[CLS] '):
    # Define the tokenizers and tokenize the strings.
    if type(strings) == list:
      strings = tf.constant(strings)

    ws_tokens, ws_start, ws_end = self.basic_tokenizer.tokenize_with_offsets(strings)
    wp_tokens, wp_start, wp_end = self.tokenizer.tokenize_with_offsets(ws_tokens)

    return {
      'ws_tokens': ws_tokens,
      'ws_start': ws_start,
      'ws_end': ws_end,
      'wp_tokens': wp_tokens,
      'wp_start': wp_start,
      'wp_end': wp_end,
    }

  @tf.function(experimental_relax_shapes=True)
  def normalize_shape(self, tokens):
    output_shape = tokens.shape[:-2].concatenate(
      tf.TensorShape([self.max_seq_length]))
    # Collapse the ragged tensor dimension by one convert to a regular tensor.
    tokens = self._merge_dims(tokens, -2)

    tokens = tokens.to_tensor(default_value='' if self.token_out_type is tf.string else 0)
    rank = len(tokens.shape)

    # Slice off some of the dim if it's too long or pad if it's too short.
    tokens = tokens[..., :self.max_seq_length]
    seq_len = tf.shape(tokens)[-1]
    paddings = [[0, 0]] * (rank - 1) + [[0, self.max_seq_length - seq_len]]

    constant_values = '' if self.token_out_type == tf.string else 0
    tokens = tf.pad(
      tokens, paddings, 'CONSTANT', constant_values=constant_values
    )

    tokens = tf.ensure_shape(tokens, output_shape)
    return tokens

  def _merge_dims(self, rt, axis=0):
    """Collapses the specified axis of a RaggedTensor.

    Suppose we have a RaggedTensor like this:
    [[1, 2, 3],
     [4, 5],
     [6]]

    If we flatten the 0th dimension, it becomes:
    [1, 2, 3, 4, 5, 6]

    Paramters
    ---------
      rt: a RaggedTensor.
      axis: the dimension to flatten.

    Returns
    -------
      A flattened RaggedTensor, which now has one less dimension.

    """
    to_expand = rt.nested_row_lengths()[axis]
    to_elim = rt.nested_row_lengths()[axis + 1]

    bar = tf.RaggedTensor.from_row_lengths(to_elim, row_lengths=to_expand)
    new_row_lengths = tf.reduce_sum(bar, axis=axis + 1)
    return tf.RaggedTensor.from_nested_row_lengths(
      rt.flat_values,
      rt.nested_row_lengths()[:axis] + (new_row_lengths,))

  def ids_to_strings(self, ids):
    return tf.gather(self.vocab, ids)


class BertEncoderInputs(tf.keras.layers.Layer):
  def call(self, tokens, training=False, int_dtype=tf.int64, **kwargs):
    input_mask = tf.equal(tokens, 0)
    input_mask = tf.cast(tf.logical_not(input_mask), int_dtype)
    input_type_ids = tf.zeros_like(tokens)

    encoder_inputs = {
      'input_word_ids': tokens,
      'input_mask': input_mask,
      'input_type_ids': input_type_ids
    }

    return encoder_inputs


class BertEmbedder(tf.keras.layers.Layer):
  def __init__(
    self,
    bert_dir: Optional[Text],
    bert_trainable: Optional[bool] = False,
    max_seq_length: Optional[int] = 128,
    float_type: Optional[Any] = tf.float32,
    **kwargs
    ):
    super(BertEmbedder, self).__init__(**kwargs)
    self.config_path = os.path.join(bert_dir, 'bert_config.json')
    self.checkpoint_dir = os.path.join(bert_dir, 'bert_checkpoint/')

    self.bert_tokenizer = BertTokenizer(
      bert_dir,
      max_seq_length
    )
    self.bert_encoder_inputs = BertEncoderInputs()
    self.bert_layer = modeling.BertModel(
      config=modeling.BertConfig.from_json_file(self.config_path),
      float_type=float_type
    )

  def call(self, strings, training=None, **kwargs):
    tokens = self.bert_tokenizer(strings)
    encoder_inputs = self.bert_encoder_inputs(tokens)

    # NOTE: CAUTION!! This does not seem to properly load, at least when using
    # in Estimator training.
    checkpoint = tf.train.Checkpoint(root=self.bert_layer, bert_layer=self.bert_layer)
    checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir)).assert_existing_objects_matched()

    embeddings, _ = self.bert_layer(encoder_inputs, training)

    return embeddings
