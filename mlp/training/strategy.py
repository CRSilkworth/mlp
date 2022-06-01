import tensorflow as tf


def choose_strategy(num_gpus: int) -> tf.distribute.Strategy:
  """Chooses the distribution strategy based on the number of gpus used."""
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      # Currently, memory growth needs to be the same across GPUs
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      # Memory growth must be set before GPUs have been initialized
      print(e)

  # How to split up over multiple gpus
  if num_gpus == 0:
    strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')
  elif num_gpus == 1:
    strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')
  elif num_gpus == 2:
    strategy = tf.distribute.MirroredStrategy()
  elif num_gpus > 2:
    strategy = tf.distribute.MirroredStrategy(
      cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
    )
  else:
    strategy = None

  return strategy
