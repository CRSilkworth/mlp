import tensorflow as tf
from typing import Optional, Dict, List, Text, Callable, Tuple


def number_to_digits(
  number: tf.Tensor,
  max_digits: Optional[int] = 12,
  cutoff: Optional[float] = 1e-10
  ) -> Tuple[tf.Tensor, tf.Tensor]:
  """
  Convert a tensor of floats to it's digit representation.
  e.g.
    number = [111.2, 0.002, 1.2]
    ->
    digits = [[1, 1, 1, 2], [2, 0, 0, 0], [1, 2, 0, 0]]
    order_of_mag = [2, -3, 0]

  Parameters
  ----------
  number: The tensor of floats to be converted.
  max_digits: The number of digits to save.
  cutoff: The minimum value below which is considered zero.

  Returns
  -------
  digits: The digit representation of the float.
  order_of_mag: The order of magnitude of the float.

  """
  is_zero = tf.math.abs(number) < cutoff
  order_of_mag = tf.math.floor(
    tf.math.log(number + tf.cast(is_zero, dtype=tf.float32)) / tf.math.log(10.)
  )
  powers = tf.math.pow(10., tf.cast(max_digits - order_of_mag - 1, dtype=tf.float32))

  integer = powers * number
  integer = tf.cast(integer, dtype=tf.int64)
  digits = integer_to_digits(integer, max_digits)

  return digits, tf.cast(order_of_mag, dtype=tf.int64)


def digits_to_number(
  digits: tf.Tensor,
  order_of_mag: tf.Tensor,
  max_digits: Optional[int] = 12
  ) -> tf.Tensor:
  """
  Convert a a digit representation to its regular float representation.
  e.g.
    digits = [[1, 1, 1, 2], [2, 0, 0, 0], [1, 2, 0, 0]]
    order_of_mag = [2, -3, 0]
    ->
    number = [111.2, 0.002, 1.2]


  Parameters
  ----------
  digits: The digit representation of the float.
  order_of_mag: The order of magnitude of the float.
  max_digits: The number of digits to save.

  Returns
  -------
  number: The float representation of the number.

  """
  powers = tf.math.pow(10., - tf.cast(max_digits - order_of_mag - 1, dtype=tf.float32))

  integer = digits_to_integers(digits)
  integer = tf.cast(integer, dtype=tf.float32)
  number = integer * powers
  return number


def integer_to_digits(
  integer: tf.Tensor,
  max_digits: Optional[int] = 12
  ) -> tf.Tensor:
  """
  Convert a tensor of integer to it's digit representation.
  e.g.
    number = [1112, 200, 12]
    -> digits = [[1, 1, 1, 2], [0, 2, 0, 0], [0, 0, 1, 2]]

  Parameters
  ----------
  integer: The tensor of integer to be converted.
  max_digits: The number of digits to save.
  cutoff: The minimum value below which is considered zero.

  Returns
  -------
  digits: The digit representation of the float.

  """
  shape = tf.shape(integer)
  powers = tf.range(max_digits - 1, -1, -1, dtype=tf.int32)
  powers = tf.math.pow(10, powers)

  multiples = tf.concat([tf.ones_like(shape), [max_digits]], axis=0)
  powers = tf.reshape(powers, multiples)
  powers = tf.tile(
    powers,
    multiples=tf.concat([shape, [1]], axis=0)
  )

  integer = tf.tile(
    tf.expand_dims(integer, axis=-1),
    multiples=multiples
  )
  powers = tf.cast(powers, dtype=tf.int64)
  digits = tf.math.floordiv(integer, powers)
  digits = tf.math.floormod(digits, 10)
  return digits


def digits_to_integers(digits: tf.Tensor) -> tf.Tensor:
  """
  Convert a tensor of digits to its integer representation.
  e.g.
    digits = [[1, 1, 1, 2], [0, 2, 0, 0], [0, 0, 1, 2]] ->
    number = [1112, 200, 12]

  Parameters
  ----------
  digits: The digit representation of the float.

  Returns
  -------
  integer: The tensor of integer to be converted.

  """
  shape = tf.shape(digits)
  max_digits = shape[-1]

  powers = tf.range(max_digits - 1, -1, -1, dtype=tf.int32)
  powers = tf.math.pow(10, powers)
  reshape = tf.concat([tf.ones_like(shape)[:-1], [max_digits]], axis=0)

  powers = tf.reshape(powers, reshape)

  multiples = tf.concat([shape[:-1], [1]], axis=0)
  powers = tf.tile(
    powers,
    multiples=multiples
  )

  powers = tf.cast(powers, dtype=tf.int64)
  integer = tf.reduce_sum(powers * digits, axis=-1)

  return integer


def digits_to_one_hots(digits: tf.Tensor) -> tf.Tensor:
  """
  Convert a tensor of digits representation to it's one hot representation.
  e.g.
    digits = [[1, 2], [2, 0], [0, 2]] ->
    [
      [
        [0, 1, ...],
        [0, 0, 1, ...]
      ],
      [
        [0, 0, 1, ...]
        [1, 0, ...],
      ],
      [
        [1, 0, ...],
        [0, 0, 1, ...]
      ],
    ]

  Parameters
  ----------
  digits: The digit representation of the number.

  Returns
  -------
  one_hots: The one hotted representation.

  """
  return tf.one_hot(digits, 10, dtype=tf.float32)


def one_hots_to_digits(one_hots: tf.Tensor) -> tf.Tensor:
  """
  Convert a tensor of digits representation to it's one hot representation.
  e.g.
    one_hots = [
      [
        [0, 1, ...],
        [0, 0, 1, ...]
      ],
      [
        [0, 0, 1, ...]
        [1, 0, ...],
      ],
      [
        [1, 0, ...],
        [0, 0, 1, ...]
      ],
    ]->
    digits = [[1, 2], [2, 0], [0, 2]]

  Parameters
  ----------
  one_hots: The one hotted representation.

  Returns
  -------
  digits: The digit representation of the number.

  """
  _, indices = tf.math.top_k(one_hots, k=1)
  digits = indices[..., 0]
  digits = tf.cast(digits, tf.int64)
  return digits


def order_of_mag_indices_to_one_hots(
  order_of_mag_indices: tf.Tensor,
  vocabulary: tf.Tensor
  ) -> tf.Tensor:
  """
  Convert a tensor of order of magnitudes indices (not actual values. Defined by the corresponding vocabulary file) to its one hot representation.
  e.g.
    digits = [3, 0, 1, 8] ->
    one_hots = [
      [0, 0, 1, ...],
      [1, 0, 0, ...],
      [0, 1, 0, ...],
      [0, 0, 0, ...]
    ]

  Parameters
  ----------
  order_of_mag_indices: The indices (not actual values) of the corresponding order of magnitudes. Should match the corresponding vocabulary.
  vocabulary: A vocab tensor of all the seen order of magnitudes.

  Returns
  -------
  one_hots: The one hotted representation.

  """
  one_hots = tf.one_hot(
    order_of_mag_indices,
    len(vocabulary),
    dtype=tf.float32
  )
  return one_hots


def one_hots_to_order_of_mag(
  one_hots: tf.Tensor,
  vocabulary: tf.Tensor
  ) -> tf.Tensor:
  """
  Convert a tensor one hotted order of magnitudes (as defined by the corresponding vocabulary file) the original order of magnitude values..
  e.g.
    one_hots = [
      [0, 0, 1, ...],
      [1, 0, 0, ...],
      [0, 1, 0, ...],
      [0, 0, 0, ...]
    ] ->
    digits = [-2, 3, -10, 0]

  Parameters
  ----------
  one_hots: The one hotted representation.
  vocabulary: A vocab tensor of all the seen order of magnitudes.

  Returns
  -------
  order_of_mag: The order of magnitudes (actual not indices).

  """
  probs, indices = tf.math.top_k(one_hots, k=1)
  order_of_mag = tf.gather(vocabulary, indices)
  order_of_mag = tf.squeeze(order_of_mag, axis=-1)
  order_of_mag = tf.strings.to_number(order_of_mag, out_type=tf.int64)
  return order_of_mag
