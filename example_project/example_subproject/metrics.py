"""Metrics definition to be fed to estimator for evaluation."""
import tensorflow as tf


def metric_fn(label_ids, probs, top_ks, learning_rate):
  """Define the standard evaluation metrics for the classifier."""
  metrics = {}
  metrics['learning_rate'] = tf.keras.metrics.Mean()
  metrics['learning_rate'].update_state(learning_rate)

  for top_k in top_ks:
    # Find whether or not the label is in the top_k predicted labels
    accuracy = tf.math.in_top_k(
      predictions=probs,
      targets=label_ids,
      k=top_k)

    # Take the mean accuracy over the dataset
    running_mean = tf.keras.metrics.Mean()
    running_mean.update_state(accuracy)

    metrics['top_' + str(top_k) + '_accuracy'] = running_mean

  return metrics
