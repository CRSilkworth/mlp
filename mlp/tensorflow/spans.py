import tensorflow as tf
import tensorflow_text as tf_text
from tensorflow.python.lib.io import file_io
from typing import Optional, Text, Tuple


@tf.function(experimental_relax_shapes=True)
def char_to_wordpiece_spans(
  ws_tokens: tf.Tensor,
  ws_start: tf.Tensor,
  ws_end: tf.Tensor,
  wp_tokens: tf.Tensor,
  wp_start: tf.Tensor,
  wp_end: tf.Tensor,
  char_start: tf.Tensor,
  char_end: tf.Tensor,
  max_seq_length: Optional[tf.Tensor] = tf.constant(512, dtype=tf.int64)) -> Tuple[tf.Tensor, tf.Tensor]:
  """Convert character span indices into wordpiece span indices. NOTE: Assumes that the span of text has no leading/trailing whitespace. If these are included, it will grab other wordpiece tokens in the wordpiece span.

  Parameters
  ----------
    ws_tokens: the tokens returned by the WhitespaceTokenizer.
    ws_start: The starts of the spans after being passed to the WhitespaceTokenizer.tokenize_with_offsets
    ws_end: The ends of the spans after being passed to the WhitespaceTokenizer.tokenize_with_offsets
    wp_tokens: the tokens returned by the WordpieceTokenizer (after being passed to the WhitespaceTokenizer).
    wp_start: The starts of the spans after being passed to the WordpieceTokenizer.tokenize_with_offsets (and before that the WhitespaceTokenizer)
    wp_start: The ends of the spans after being passed to the WordpieceTokenizer.tokenize_with_offsets (and before that the WhitespaceTokenizer)
    char_start: The starts of the character indexed spans.
    char_start: The ends of the character indexed spans.
    max_seq_length: The maximum allowed number of wordpiece tokens

  Returns
  -------
  new_span_starts: The span beginnings in the word piece indices
  new_span_ends: The span ends in the word piece indices
  """
  # Cast all integers to int64.
  ws_start, ws_end = tf.cast(ws_start, tf.int64), tf.cast(ws_end, tf.int64)
  wp_start, wp_end = tf.cast(wp_start, tf.int64), tf.cast(wp_end, tf.int64)
  word_piece_shape = tf.cast(wp_tokens.bounding_shape(), tf.int64)
  span_shape = tf.cast(tf.shape(char_start), tf.int64)

  batch_size = tf.cast(word_piece_shape[0], tf.int32)
  max_words = tf.cast(word_piece_shape[1], tf.int32)
  max_pieces = tf.cast(word_piece_shape[2], tf.int32)
  num_spans = tf.cast(span_shape[1], tf.int32)

  new_span_starts = tf.TensorArray(tf.int64, batch_size * num_spans)
  new_span_ends = tf.TensorArray(tf.int64, batch_size * num_spans)

  # Create a counter to find the total word piece index from the (word index,
  # word piece index) combination
  ones = tf.ones_like(wp_tokens, dtype=tf.int64)
  word_pieces_in_words = tf.reduce_sum(ones, axis=2)

  # Iterate through each row in the batch
  for row_num in tf.range(batch_size, dtype=tf.int64):

    words = ws_tokens[row_num]
    num_words = tf.cast(tf.shape(words)[0], tf.int64)

    # Iterate through each word
    for word_num in tf.range(max_words, dtype=tf.int64):

      # Since these are ragged tensors, must check it's a valid word_num.
      if word_num < num_words:
        word_pieces = wp_tokens[row_num][word_num]
        word_start = ws_start[row_num][word_num]
        word_piece_start = tf.reduce_sum(word_pieces_in_words[row_num, :word_num])
        num_pieces = tf.cast(tf.shape(word_pieces)[0], tf.int64)

        # Iterate through each word piece
        for piece_num in tf.range(max_pieces, dtype=tf.int64):

          # Since these are ragged tensors, must check it's a valid piece_num.
          if piece_num < num_pieces:
            piece_start = wp_start[row_num][word_num][piece_num]
            piece_end = wp_end[row_num][word_num][piece_num]
            word_piece_num = word_piece_start + piece_num

            for span_num in tf.range(num_spans, dtype=tf.int64):
              span_start = tf.cast(char_start[row_num][span_num], tf.int64)
              span_end = tf.cast(char_end[row_num][span_num], tf.int64)
              # Take the last word piece num such that the span does not start in
              # the middle of the word piece, then add one. This guards against
              # the case someone included whitespace in the span.
              if span_start >= word_start + piece_end:
                index = tf.cast(row_num * tf.cast(num_spans, tf.int64) + span_num, tf.int32)
                new_span_starts = new_span_starts.write(index, word_piece_num + 1)
              if span_end > word_start + piece_start:
                index = tf.cast(row_num * tf.cast(num_spans, tf.int64) + span_num, tf.int32)
                new_span_ends = new_span_ends.write(index, word_piece_num + 1)

  new_span_starts = new_span_starts.stack()
  new_span_ends = new_span_ends.stack()

  new_span_starts = tf.reshape(new_span_starts, [batch_size, num_spans])
  new_span_ends = tf.reshape(new_span_ends, [batch_size, num_spans])

  # Truncate any indices that go over the max sequence length.
  new_span_starts = tf.where(
    new_span_starts < max_seq_length,
    new_span_starts,
    tf.cast(max_seq_length, tf.int64)
  )

  new_span_ends = tf.where(
    new_span_ends > max_seq_length,
    tf.cast(max_seq_length, tf.int64),
    new_span_ends,
  )
  return new_span_starts, new_span_ends


# def char_to_wordpiece_spans(
#   strings: tf.Tensor,
#   span_starts: tf.Tensor,
#   span_ends: tf.Tensor,
#   vocab_file_path: Text,
#   max_seq_length: Optional[int] = 512,
#   begin_token: Optional[Text] = '[CLS] ',
#   ) -> (tf.Tensor, tf.Tensor):
#   """Convert character span indices into wordpiece span indices. NOTE: Assumes that the span of text has no leading/trailing whitespace. If these are included, it will grab other wordpiece tokens in the wordpiece span.
#
#   Parameters
#   ----------
#     strings: The text which will be tokenized by whitespace tokenizer then wordpiece tokenizer. It's shape should be [batch_size].
#     span_starts: The start of the spans. Should be integer valued with shape [batch_size]. NOTE: Assumes that the span of text has no leading whitespace.
#     span_ends: The end of the spans. Should be integer valued with shape [batch_size]. NOTE: Assumes that the span of text has no trailing .
#     vocab_file_path: The path to the wordpiece vocabulary definition.
#
#   Returns
#   -------
#   new_span_starts: The span beginnings in the word piece indices
#   new_span_ends: The span ends in the word piece indices
#   """
#   # Correct the character span for the added character from begin_token
#   span_starts = span_starts + len(begin_token)
#   span_ends = span_ends + len(begin_token)
#
#   # Define the vocabulary table to be used in the wordpiece tokenization.
#   with file_io.FileIO(vocab_file_path, mode='r') as vocab_file:
#     vocab = vocab_file.read().split()
#   init = tf.lookup.KeyValueTensorInitializer(
#     vocab,
#     tf.range(tf.size(vocab, out_type=tf.int64), dtype=tf.int64),
#     key_dtype=tf.string,
#     value_dtype=tf.int64
#   )
#   vocab_table = tf.lookup.StaticVocabularyTable(
#     init, 1, lookup_key_dtype=tf.string
#   )
#
#   # Define the tokenizers
#   whitespace_tokenizer = tf_text.WhitespaceTokenizer()
#   tokenizer = tf_text.WordpieceTokenizer(
#     vocab_table,
#     token_out_type=tf.string
#   )
#
#   # Tokenize the strings, keeping the indices where each token starts and ends.
#   ws_tokens, ws_start, ws_end = whitespace_tokenizer.tokenize_with_offsets(begin_token + strings)
#   wp_tokens, wp_start, wp_end = tokenizer.tokenize_with_offsets(ws_tokens)
#
#   # Cast all integers to int64.
#   ws_start, ws_end = tf.cast(ws_start, tf.int64), tf.cast(ws_end, tf.int64)
#   wp_start, wp_end = tf.cast(wp_start, tf.int64), tf.cast(wp_end, tf.int64)
#   word_piece_shape = tf.cast(wp_tokens.bounding_shape(), tf.int64)
#   span_shape = tf.cast(tf.shape(span_starts), tf.int64)
#
#   batch_size = tf.cast(word_piece_shape[0], tf.int32)
#   max_words = tf.cast(word_piece_shape[1], tf.int32)
#   max_pieces = tf.cast(word_piece_shape[2], tf.int32)
#   num_spans = tf.cast(span_shape[1], tf.int32)
#
#   new_span_starts = tf.TensorArray(tf.int64, batch_size * num_spans)
#   new_span_ends = tf.TensorArray(tf.int64, batch_size * num_spans)
#
#   # Create a counter to find the total word piece index from the (word index,
#   # word piece index) combination
#   ones = tf.ones_like(wp_tokens, dtype=tf.int64)
#   word_pieces_in_words = tf.reduce_sum(ones, axis=2)
#
#   # Iterate through each row in the batch
#   for row_num in tf.range(batch_size, dtype=tf.int64):
#
#     words = ws_tokens[row_num]
#     num_words = tf.cast(tf.shape(words)[0], tf.int64)
#
#     # Iterate through each word
#     for word_num in tf.range(max_words, dtype=tf.int64):
#
#       # Since these are ragged tensors, must check it's a valid word_num.
#       if word_num < num_words:
#         word_pieces = wp_tokens[row_num][word_num]
#         word_start = ws_start[row_num][word_num]
#         word_piece_start = tf.reduce_sum(word_pieces_in_words[row_num, :word_num])
#         num_pieces = tf.cast(tf.shape(word_pieces)[0], tf.int64)
#
#         # Iterate through each word piece
#         for piece_num in tf.range(max_pieces, dtype=tf.int64):
#
#           # Since these are ragged tensors, must check it's a valid piece_num.
#           if piece_num < num_pieces:
#             piece_start = wp_start[row_num][word_num][piece_num]
#             piece_end = wp_end[row_num][word_num][piece_num]
#             word_piece_num = word_piece_start + piece_num
#
#             for span_num in tf.range(num_spans, dtype=tf.int64):
#               span_start = tf.cast(span_starts[row_num][span_num], tf.int64)
#               span_end = tf.cast(span_ends[row_num][span_num], tf.int64)
#               # Take the last word piece num such that the span does not start in
#               # the middle of the word piece, then add one. This guards against
#               # the case someone included whitespace in the span.
#               if span_start >= word_start + piece_end:
#                 index = tf.cast(row_num * tf.cast(num_spans, tf.int64) + span_num, tf.int32)
#                 new_span_starts = new_span_starts.write(index, word_piece_num + 1)
#               if span_end > word_start + piece_start:
#                 index = tf.cast(row_num * tf.cast(num_spans, tf.int64) + span_num, tf.int32)
#                 new_span_ends = new_span_ends.write(index, word_piece_num + 1)
#
#   new_span_starts = new_span_starts.stack()
#   new_span_ends = new_span_ends.stack()
#
#   new_span_starts = tf.reshape(new_span_starts, [batch_size, num_spans])
#   new_span_ends = tf.reshape(new_span_ends, [batch_size, num_spans])
#
#   # Truncate any indices that go over the max sequence length.
#   new_span_starts = tf.where(
#     new_span_starts < max_seq_length,
#     new_span_starts,
#     tf.cast(max_seq_length, tf.int64)
#   )
#
#   new_span_ends = tf.where(
#     new_span_ends > max_seq_length,
#     tf.cast(max_seq_length, tf.int64),
#     new_span_ends,
#   )
#
#   return new_span_starts, new_span_ends
