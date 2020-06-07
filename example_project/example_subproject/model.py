"""Model definition for the BasicNN model."""
import tensorflow as tf
from typing import Optional, List


class BasicNN(tf.keras.Model):
  """Basic classifier neural network model."""

  def __init__(
    self,
    hidden_layer_dims: List[int],
    num_labels: int,
    dropout_rate: Optional[float] = 0.7,
    ):
    """Construct an BasicNN model.

    Parameters
    ----------
    num_labels: The total number of allowed labels.
    hidden_layer_dims: The dimensions of the dense layers that combine the
      non_string_key data and output from the bert model.

    """
    super(BasicNN, self).__init__()
    self.hidden_layer_dims = hidden_layer_dims
    self.num_labels = num_labels
    self.dropout_rate = dropout_rate
    self.classifier_weights = []

    initializer = tf.keras.initializers.TruncatedNormal()

    # Define the dropout and dense layers that will appear after the Bert layers
    self.dropout_layers = []
    self.dense_layers = []
    for layer_num, layer_dim in enumerate(hidden_layer_dims + [num_labels]):
      dropout_layer = tf.keras.layers.Dropout(
        rate=self.dropout_rate
      )
      dense_layer = tf.keras.layers.Dense(
        layer_dim,
        kernel_initializer=initializer,
        name='output',
        activation='relu' if layer_num != len(hidden_layer_dims) else None,
        dtype=tf.float32
      )

      # Hold on to the layer objects and their weights.
      self.classifier_weights.extend(dense_layer.trainable_variables)
      self.dropout_layers.append(dropout_layer)
      self.dense_layers.append(dense_layer)

  def call(self, inputs, training=False, **kwargs):
    """Get the logits predicted by the model.

    Parameters
    ----------
    inputs: will be ignored. Only there to conform to keras model standard.
    traininig: Whether or not the model is training.
    kwargs: Any data to be used in the prediction.

    Returns
    -------
    The logits predicted by the model.

    """
    all_inputs = [kwargs[s] for s in kwargs]
    logits = tf.concat(all_inputs, axis=1)

    # Send the logits + non_string_keys through the classifier weights
    for layer_num, layer_dim in enumerate(self.hidden_layer_dims + [self.num_labels]):
      logits = self.dropout_layers[layer_num](logits, training=training)
      logits = self.dense_layers[layer_num](logits)

    return logits
