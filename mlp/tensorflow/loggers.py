import tensorflow as tf
import absl
from timeit import default_timer
from tensorflow.python.profiler import profiler_v2 as profiler
from tensorflow.python.framework import errors

class NBatchLogger(tf.keras.callbacks.Callback):
  def __init__(self, display):
    self.step = 0
    self.display = display

  def on_epoch_begin(self, epoch, logs=None):
    super(NBatchLogger, self).on_epoch_begin(epoch, logs)
    self.step = 0
    self.epoch_start_time = default_timer()

  def on_train_batch_end(self, batch, logs=None):
    logs = logs or {}
    self.step += 1

    if self.step % self.display == 0:

      av_step_time = (default_timer() - self.epoch_start_time) / self.step
      metrics_log = 'sec/steps: %.4f' % av_step_time

      for k in logs:
        val = logs[k]
        if abs(val) > 1e-3:
            metrics_log += ' - %s: %.4f' % (k, val)
        else:
            metrics_log += ' - %s: %.4e' % (k, val)

      print_string = 'step: {}/{} ... {}'.format(
        self.step,
        self.params['steps'],
        metrics_log
        )
      absl.logging.info(print_string)


class TensorBoardWithOptionsCallback(tf.keras.callbacks.TensorBoard):
  def __init__(
    self,
    **kwargs
    ):
    self.options = kwargs['options']
    del kwargs['options']
    super(TensorBoardWithOptionsCallback, self).__init__(**kwargs)

  def _start_profiler(self, logdir):
    """Starts the profiler if currently inactive.
    Args:
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
