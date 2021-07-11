import tensorflow as tf


def piecewise_learning_rate(global_step, learning_rate, num_train_steps, num_warmup_steps, num_cooldown_steps, warmup_power=1.0, cooldown_power=1.0, name=None):
  if num_warmup_steps + num_cooldown_steps > num_train_steps:
    raise ValueError("Number of train steps must be greater than or equal to the number of warmup steps plus cool down steps. Got {} + {} and {}".format(num_warmup_steps, num_cooldown_steps, num_train_steps))

  global_steps_int = tf.cast(global_step, tf.int32)
  train_steps_int = tf.constant(num_train_steps, dtype=tf.int32)
  warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)
  cooldown_steps_int = tf.constant(num_cooldown_steps, dtype=tf.int32)

  global_steps_float = tf.cast(global_steps_int, tf.float32)
  train_steps_float = tf.cast(train_steps_int, tf.float32)
  warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)
  cooldown_steps_float = tf.cast(cooldown_steps_int, tf.float32)
  steps_into_cooldown = global_steps_float - (train_steps_float - cooldown_steps_float)

  is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
  is_middle = tf.cast((global_steps_int >= warmup_steps_int) & (steps_into_cooldown <= 0), tf.float32)
  is_cooldown = tf.cast(steps_into_cooldown > 0, tf.float32)

  if num_warmup_steps == 0:
    warmup_learning_rate = 0.0
  else:
    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = learning_rate * tf.math.pow(warmup_percent_done, warmup_power)

  if num_cooldown_steps == 0:
    cooldown_learning_rate = 0.0
  else:
    cooldown_percent_done = steps_into_cooldown / cooldown_steps_float
    cooldown_learning_rate = learning_rate * (1.0 - tf.math.pow(is_cooldown * cooldown_percent_done, cooldown_power))

  adj_learning_rate = (
    is_warmup * warmup_learning_rate +
    is_middle * learning_rate +
    is_cooldown * cooldown_learning_rate
  )
  if name is not None:
    adj_learning_rate = tf.identity(adj_learning_rate, name=name)
  return adj_learning_rate


class PiecewiseLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, learning_rate, num_train_steps, warmup_prop=1.0, cooldown_prop=1.0, warmup_power=1.0, cooldown_power=1.0):
    super(PiecewiseLearningRate, self).__init__()

    self.learning_rate = learning_rate
    self.num_train_steps = num_train_steps
    self.warmup_prop = warmup_prop
    self.cooldown_prop = cooldown_prop
    self.warmup_power = warmup_power
    self.cooldown_power = cooldown_power

  def __call__(self, step):
    num_warmup_steps = int(self.num_train_steps * self.warmup_prop)
    num_cooldown_steps = int(self.num_train_steps * self.cooldown_prop)

    return piecewise_learning_rate(
      global_step=step,
      learning_rate=self.learning_rate,
      num_train_steps=self.num_train_steps,
      num_warmup_steps=num_warmup_steps,
      num_cooldown_steps=num_cooldown_steps,
      warmup_power=self.warmup_power,
      cooldown_power=self.cooldown_power
    )

  def get_config(self):
    config = {
      'learning_rate': self.learning_rate,
      'num_train_steps': self.num_train_steps,
      'warmup_prop': self.warmup_prop,
      'cooldown_prop': self.cooldown_prop,
      'warmup_power': self.warmup_power,
      'cooldown_power': self.cooldown_power,
     }
    return config
