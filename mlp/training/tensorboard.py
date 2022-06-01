import tensorflow as tf
from tensorflow.python.profiler import profiler_v2 as profiler
from tensorflow.python.framework import errors
import absl


class TensorBoardWithOptionsCallback(tf.keras.callbacks.TensorBoard):
  """Allows for slightly modified tensorboard. e.g. with profiler."""

  def __init__(
    self,
    **kwargs
    ):
    """Modify the options and then call the parent class init."""
    self.options = kwargs['options']
    del kwargs['options']
    super(TensorBoardWithOptionsCallback, self).__init__(**kwargs)

  def _start_profiler(self, logdir):
    """Start the profiler if currently inactive.

    Parameters
    ----------
    logdir: Directory where profiler results will be saved.

    """
    if self._profiler_started:
      return
    try:
      profiler.start(logdir=logdir, options=self.options)
      self._profiler_started = True
    except errors.AlreadyExistsError as e:
      # Profiler errors should not be fatal.
      absl.logging.error('Failed to start profiler: %s', e.message)
