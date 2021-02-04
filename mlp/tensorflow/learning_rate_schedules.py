import tensorflow as tf


def piecewise_learning_rate(global_step, learning_rate, num_train_steps, num_warmup_steps, num_cool_down_steps, warmup_power=1.0, cool_down_power=1.0):
  if num_warmup_steps + num_cool_down_steps > num_train_steps:
    raise ValueError("Number of train steps must be greater than or equal to the number of warmup steps plus cool down steps. Got {} + {} and {}".format(num_warmup_steps, num_cool_down_steps, num_train_steps))

  global_steps_int = tf.cast(global_step, tf.int32)
  train_steps_int = tf.constant(num_train_steps, dtype=tf.int32)
  warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)
  cool_down_steps_int = tf.constant(num_cool_down_steps, dtype=tf.int32)

  global_steps_float = tf.cast(global_steps_int, tf.float32)
  train_steps_float = tf.cast(train_steps_int, tf.float32)
  warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)
  cool_down_steps_float = tf.cast(cool_down_steps_int, tf.float32)
  steps_into_cool_down = global_steps_float - (train_steps_float - cool_down_steps_float)

  is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
  is_middle = tf.cast((global_steps_int >= warmup_steps_int) & (steps_into_cool_down <= 0), tf.float32)
  is_cool_down = tf.cast(steps_into_cool_down > 0, tf.float32)

  if num_warmup_steps == 0:
    warmup_learning_rate = 0.0
  else:
    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = learning_rate * tf.math.pow(warmup_percent_done, warmup_power)

  if num_cool_down_steps == 0:
    cool_down_learning_rate = 0.0
  else:
    cool_down_percent_done = steps_into_cool_down / cool_down_steps_float
    cool_down_learning_rate = learning_rate * (1.0 - tf.math.pow(is_cool_down * cool_down_percent_done, cool_down_power))

  adj_learning_rate = (
    is_warmup * warmup_learning_rate +
    is_middle * learning_rate +
    is_cool_down * cool_down_learning_rate
  )
  
  return adj_learning_rate
