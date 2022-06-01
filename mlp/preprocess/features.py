import tensorflow as tf


def feature_from_scalar(value: tf.Tensor, batch_size: int) -> tf.Tensor:
  """Create a featurer of the proper batch size from a scalar value."""
  return tf.tile(tf.expand_dims(value, 0), multiples=[batch_size])
