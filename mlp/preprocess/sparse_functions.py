from typing import Optional, Text, List, Any
import tensorflow as tf


def sparse_to_dense_with_fill(
  x: tf.Tensor,
  expected_dtype: Optional[tf.DType] = None,
  default_value: Optional[Any] = None) -> tf.Tensor:
  """Replace missing values in a SparseTensor.

  Fills in missing values of `x` with '' or 0, and converts to a dense tensor.

  Parameters
  ----------
    x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
      in the second dimension.

  Returns
  -------
    A rank 1 tensor where missing values of `x` have been filled in.

  """
  values = x.values
  indices = x.indices
  if expected_dtype == tf.int64 and x.dtype == tf.string:
    values = tf.strings.to_number(values, tf.int64)

  if default_value is None:
    if expected_dtype == tf.int64 and x.dtype == tf.string:
      default_value = 0
    else:
      default_value = '' if x.dtype == tf.string else 0

  return tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(indices, values, [x.dense_shape[0], 1]),
          default_value),
      axis=1)
