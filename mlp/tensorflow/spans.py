import tensorflow as tf
from typing import Optional, Text, Tuple, List, Callable
from mlp.layers.bert import BertTokenizer
from tensorflow_text.python.ops.bert_tokenizer import BasicTokenizer


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
    wp_end: The ends of the spans after being passed to the WordpieceTokenizer.tokenize_with_offsets (and before that the WhitespaceTokenizer)
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


@tf.function(experimental_relax_shapes=True)
def wordpiece_to_char_spans(
  ws_start: tf.Tensor,
  ws_end: tf.Tensor,
  wp_start: tf.Tensor,
  wp_end: tf.Tensor,
  span_start: tf.Tensor,
  span_end: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
  """Convert character span indices into wordpiece span indices.

  Parameters
  ----------
    ws_start: The starts of the spans after being passed to the WhitespaceTokenizer.tokenize_with_offsets
    ws_end: The ends of the spans after being passed to the WhitespaceTokenizer.tokenize_with_offsets
    wp_start: The starts of the spans after being passed to the WordpieceTokenizer.tokenize_with_offsets (and before that the WhitespaceTokenizer)
    wp_end: The ends of the spans after being passed to the WordpieceTokenizer.tokenize_with_offsets (and before that the WhitespaceTokenizer)
    span_start: The starts of the wordpiece indexed spans.
    span_end: The ends of the wordpiece indexed spans.

  Returns
  -------
  char_start: The span beginnings in the character indices
  char_end: The span ends in the character indices

  """
  word_piece_shape = tf.cast(wp_start.bounding_shape(), tf.int32)
  max_ws_words = tf.cast(word_piece_shape[1], tf.int32)
  max_wps = tf.cast(word_piece_shape[2], tf.int32)

  span_shape = tf.cast(tf.shape(span_start), tf.int32)
  num_spans = tf.cast(span_shape[1], tf.int32)

  batch_size = tf.cast(word_piece_shape[0], tf.int32)

  new_span_start = tf.TensorArray(tf.int64, batch_size * num_spans)
  new_span_end = tf.TensorArray(tf.int64, batch_size * num_spans)

  # Iterate through each row in the batch
  for row_num in tf.range(batch_size, dtype=tf.int32):
    num_ws_words = tf.cast(wp_start[row_num].bounding_shape()[0], tf.int32)

    # Create the mapping between the word piece index from before the
    # merge_dims is performed and after.
    wp_indices = tf.range(tf.shape(wp_start[row_num].values)[0], dtype=tf.int32)
    wp_indices = tf.RaggedTensor.from_row_starts(
      wp_indices,
      wp_start[row_num].row_starts()
    )

    # Iterate through each span
    for span_num in tf.range(num_spans, dtype=tf.int32):

      # Pull out all the pan information and the array_index. i.e. where in the
      # TensorArray to place it.
      array_index = row_num * num_spans + span_num
      wp_span_start = tf.cast(span_start[row_num][span_num], dtype=tf.int32)
      wp_span_end = tf.cast(span_end[row_num][span_num], dtype=tf.int32)

      # Iterate through all the white space tokenized words.
      for ws_word_num in tf.range(max_ws_words, dtype=tf.int32):

        # Make sure it's a valid white space tokenized word.
        if ws_word_num < num_ws_words:

          # Get the number of word pieces for this row/white space word
          num_wps = tf.cast(
            tf.shape(wp_start[row_num][ws_word_num])[0],
            tf.int32
          )

          # Iterate through every wordpiece
          for wp_num in tf.range(max_wps, dtype=tf.int32):

            # Make sure it's a valid wordpiece
            if wp_num < num_wps:
              wp_index = wp_indices[ws_word_num, wp_num]

              # If the constructed word piece index (wp_index) is the same as
              # the given span_start index, then write it to the tensor array.
              if wp_span_start == wp_index:
                new_span_start = new_span_start.write(
                  array_index,
                  tf.cast(ws_start[row_num][ws_word_num], tf.int64) + wp_start[row_num][ws_word_num][wp_num]
                )

              # If the constructed word piece index (wp_index) is the same as
              # the previous span_end index, then write it to the tensor array.
              # You take the previous one because the end is not inclusive.
              if wp_span_end - 1 == wp_index:
                new_span_end = new_span_end.write(
                  array_index,
                  tf.cast(ws_start[row_num][ws_word_num], tf.int64) + wp_end[row_num][ws_word_num][wp_num]
                )

  char_start = new_span_start.stack()
  char_start = tf.reshape(char_start, [batch_size, num_spans])
  char_end = new_span_end.stack()
  char_end = tf.reshape(char_end, [batch_size, num_spans])

  # You can omit this if you want to allow invalid spans. However because of
  # the way you take the previous wp token end to define get the char_end,
  # you'll need to guard against char_end < char_start when span_start ==
  # span_end.
  char_end = tf.math.maximum(char_start, char_end)

  return char_start, char_end


@tf.function
def unicode_to_byte_indices(string, unicode_indices):

  result = tf.strings.unicode_decode_with_offsets(string, 'UTF-8')

  batch_size = tf.shape(string)[0]
  max_fields = tf.shape(unicode_indices)[1]

  batch_dim = tf.range(batch_size, dtype=tf.int64)
  batch_dim = tf.reshape(batch_dim, [batch_size, 1, 1])
  batch_dim = tf.tile(batch_dim, multiples=[1, max_fields, 1])

  unicode_indices = tf.expand_dims(unicode_indices, axis=-1)
  gather_indices = tf.concat([batch_dim, unicode_indices], axis=-1)

  byte_indices = tf.gather_nd(result[1], gather_indices)

  return byte_indices


@tf.function
def get_chat_tokens(
  user_message: tf.Tensor,
  system_response: tf.Tensor,
  chat_history: tf.Tensor,
  bert_vocab: List[Text],
  bert_dir: Optional[Text] = 'gs://biwako-prd/bert/data/',
  begin_token: Optional[Text] = '[CLS]',
  separator_token: Optional[Text] = '[SEP]',
  max_seq_length: Optional[int] = 384,
  token_out_type: Optional[tf.DType] = tf.int64,
  tokenizer: Optional[Callable] = None
  ):

  batch_size = tf.shape(user_message)[0]

  if token_out_type in (tf.int32, tf.int64):
    begin_token_id = bert_vocab.index(begin_token)
    separator_token_id = bert_vocab.index(separator_token)

    begin_token_tensor = tf.ones([batch_size, 1, 1], dtype=tf.int64) * begin_token_id
    separator_token_tensor = tf.ones([batch_size, 1, 1], dtype=tf.int64) * separator_token_id
  elif token_out_type == tf.string:
    begin_token_tensor = tf.tile(
      tf.reshape(begin_token, [1, 1, 1]),
      multiples=[batch_size, 1, 1]
    )
    separator_token_tensor = tf.tile(
      tf.reshape(separator_token, [1, 1, 1]),
      multiples=[batch_size, 1, 1]
    )
  else:
    raise ValueError("token_out_type must be either int32, int64 or string. got {} ".format(token_out_type))

  if tokenizer is None:
    tokenizer = BertTokenizer(bert_dir, max_seq_length, token_out_type=token_out_type, basic_tokenizer_class=BasicTokenizer)

  chat_tokens = []
  for message_num, message in enumerate([user_message, system_response, chat_history]):
    token_dict = tokenizer.tokens_and_spans(
      message
    )
    if message_num == 0:
      chat_tokens.append(begin_token_tensor)
      user_message_token_dict = token_dict
    else:
      chat_tokens.append(separator_token_tensor)
    chat_tokens.append(token_dict['wp_tokens'])

  chat_tokens = tf.concat(chat_tokens, axis=1)
  chat_tokens = tokenizer.normalize_shape(chat_tokens)

  return user_message_token_dict, chat_tokens


@tf.function
def adjust_fake_spans(message, is_span, fake_data, span_start, span_end, max_fields, keep_original_prob=0.1):
  batch_size = tf.shape(span_start)[0]
  # max_fields = tf.shape(span_start)[1]

  fake_data = tf.where(is_span, fake_data, '')

  batch_dim = tf.range(batch_size, dtype=tf.int32)
  batch_dim = tf.tile(
    tf.reshape(batch_dim, [batch_size, 1, 1]),
    multiples=[1, max_fields, 1]
  )
  # Sort by span_end since there may be 'real' span_starts = 1.  There are no
  # 'real' span_ends = 1, so it'll get properly sorted.
  sort_indices = tf.argsort(span_end, axis=-1)
  unsort_indices = tf.argsort(sort_indices, axis=-1)

  sort_indices = tf.expand_dims(sort_indices, axis=-1)
  sort_indices = tf.concat([batch_dim, sort_indices], axis=-1)

  unsort_indices = tf.expand_dims(unsort_indices, axis=-1)
  unsort_indices = tf.concat([batch_dim, unsort_indices], axis=-1)

  sorted_span_start = tf.cast(tf.gather_nd(span_start, sort_indices), tf.int32)
  sorted_span_end = tf.cast(tf.gather_nd(span_end, sort_indices), tf.int32)
  sorted_fake_data = tf.gather_nd(fake_data, sort_indices)

  original_spans = tf.strings.substr(
    tf.expand_dims(message, axis=-1),
    sorted_span_start,
    sorted_span_end - sorted_span_start
  )

  keep_original = tf.random.uniform(sorted_fake_data.shape, maxval=1.0, dtype=tf.float32)
  keep_original = keep_original < keep_original_prob
  keep_original = keep_original | (sorted_fake_data == b'')
  sorted_fake_data = tf.where(
    keep_original,
    original_spans,
    sorted_fake_data)

  zeros = tf.zeros([batch_size], tf.int32)
  new_message = tf.strings.substr(message, zeros, zeros)
  for field_num in range(max_fields):

    if field_num == 0:
      start_slice = tf.zeros([batch_size], dtype=tf.int32)
    else:
      start_slice = sorted_span_end[:, field_num - 1]

    lengths = sorted_span_start[:, field_num] - start_slice

    new_message += tf.strings.substr(message, start_slice, lengths) + sorted_fake_data[:, field_num]

  new_message += tf.strings.substr(message, sorted_span_end[:, field_num], -tf.ones([batch_size], dtype=tf.int32))

  fake_lens = tf.strings.length(sorted_fake_data)
  orig_lens = sorted_span_end - sorted_span_start

  diff_lens = fake_lens - orig_lens

  cumsum = tf.cumsum(diff_lens, axis=-1, exclusive=True)

  sorted_span_start = cumsum + sorted_span_start
  sorted_span_end = sorted_span_start + fake_lens

  span_start = tf.gather_nd(sorted_span_start, unsort_indices)
  span_end = tf.gather_nd(sorted_span_end, unsort_indices)

  return new_message, span_start, span_end


@tf.function
def char_to_wordpiece(span_start, span_end, tokens_dict, max_seq_length, bert_dir):
  char_start = span_start
  char_end = span_end

  new_span_start, new_span_end = char_to_wordpiece_spans(
    ws_tokens=tokens_dict['ws_tokens'],
    ws_start=tokens_dict['ws_start'],
    ws_end=tokens_dict['ws_end'],
    wp_tokens=tokens_dict['wp_tokens'],
    wp_start=tokens_dict['wp_start'],
    wp_end=tokens_dict['wp_end'],
    char_start=char_start,
    char_end=char_end,
    max_seq_length=tf.constant(max_seq_length, dtype=tf.int64)
  )

  wp_span_start = tf.ensure_shape(new_span_start, span_start.shape)
  wp_span_end = tf.ensure_shape(new_span_end, span_end.shape)

  return wp_span_start, wp_span_end
