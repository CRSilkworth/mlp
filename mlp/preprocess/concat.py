from typing import Optional, Text, List, Any
import tensorflow as tf
from mlp.preprocess.sparse_functions import sparse_to_dense_with_fill


def concat_inputs(inputs, key, num_steps, expected_dtype=None, default_value=None):
  concated = []
  for step_num in range(num_steps):
    filled = sparse_to_dense_with_fill(
      inputs[key + '_' + str(step_num)],
      expected_dtype=expected_dtype,
      default_value=default_value
    )
    filled = tf.expand_dims(filled, axis=1)
    concated.append(filled)
  concated = tf.concat(
    concated,
    axis=1
  )
  return concated


def unconcat_inputs(concated, key, num_steps):
  unconcated = tf.split(concated, num_steps, axis=-1)

  r_dict = {}
  for num in range(num_steps):
    r_dict[key + '_' + str(num)] = tf.squeeze(unconcated[num], axis=-1)

  return r_dict
