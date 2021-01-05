import tensorflow as tf
import os
import absl
import tensorflow_model_analysis as tfma
import json

from typing import Callable, Text


class ExportEvalSavedModelHook(tf.estimator.SessionRunHook):
  def __init__(
    self,
    estimator: tf.estimator.Estimator,
    eval_model_dir: Text,
    eval_input_receiver_fn: Callable):
    self.estimator = estimator
    self.eval_input_receiver_fn = eval_input_receiver_fn
    interval_eval_model_dir = '/'.join(eval_model_dir.split('/')[:-1])
    self.eval_model_dir = os.path.join(interval_eval_model_dir, 'interval_eval_model_dir')
    self.running = False

  def end(
    self,
    session: tf.compat.v1.Session):

    # export_eval_savedmodel creates a session, which can lead to this an
    # an infinite recursion. This checks if this instance is already running,
    # and only runs it if not.
    if self.running:
      return
    self.running = True

    absl.logging.info('Exporting eval_savedmodel to %s.', self.eval_model_dir)

    if self._is_chief():
      tfma.export.export_eval_savedmodel(
          estimator=self.estimator,
          export_dir_base=self.eval_model_dir,
          eval_input_receiver_fn=self.eval_input_receiver_fn)

    self.running = False

  def _is_chief(self):
    """Returns true if this is run in the master (chief) of training cluster."""
    tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')

    # If non distributed mode, current process should always behave as chief.
    if not tf_config or not tf_config.get('cluster', {}):
      return True

    task_type = tf_config['task']['type']
    task_index = tf_config['task']['index']

    # 'master' is a legacy notation of chief node in distributed training flock.
    return task_type == 'chief' or (task_type == 'master' and task_index == 0)
