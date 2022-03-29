"""Model definition for the BasicNN model."""
import tensorflow as tf
import tensorflow_transform as tft
from typing import Optional, List, Text, Dict


class BasicNN(tf.keras.Model):
  """Basic classifier neural network model."""

  def __init__(
    self,
    hidden_layer_dims: List[int],
    num_labels: int,
    vocabularies: Dict[Text, List[Text]],
    tf_transform_output_dir: Text,
    dropout_rate: Optional[float] = 0.3,
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
    self.vocabularies = vocabularies
    self.dropout_rate = dropout_rate
    self.classifier_weights = []
    self.tf_transform_output = tft.TFTransformOutput(tf_transform_output_dir)

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

  def call(self, inputs, training=False):
    """Get the logits predicted by the model.

    Parameters
    ----------
    inputs: Any data to be used in the prediction.
    traininig: Whether or not the model is training.

    Returns
    -------
    The logits predicted by the model.

    """

    all_inputs = []
    for key in inputs:
      if inputs[key].dtype == tf.int64:
        input = tf.one_hot(
          inputs[key],
          depth=len(self.vocabularies[key.replace('_xf', '')]),
          dtype=tf.float32
        )
        input = tf.squeeze(input, axis=1)
      else:
        input = inputs[key]
      all_inputs.append(input)

    logits = tf.concat(all_inputs, axis=1)

    # Send the logits + non_string_keys through the classifier weights
    for layer_num, layer_dim in enumerate(self.hidden_layer_dims + [self.num_labels]):
      logits = self.dropout_layers[layer_num](logits, training=training)
      logits = self.dense_layers[layer_num](logits)

    return logits

  def get_serve_tf_examples_fn(self, categorical_feature_keys, numerical_feature_keys, all_labels):
    """Returns a function that parses a serialized tf.Example and applies TFT."""

    self.tft_layer = self.tf_transform_output.transform_features_layer()
    feature_spec = self.tf_transform_output.raw_feature_spec()

    input_keys = categorical_feature_keys + numerical_feature_keys

    input_signature = []
    for key in input_keys:
      dtype = feature_spec[key].dtype
      input_signature.append(tf.TensorSpec([None], dtype=dtype, name=key))

    @tf.function(input_signature=input_signature)
    def serve_tf_examples_fn(*args):
      """Returns the output to be used in the serving signature."""

      receiver_tensors = {arg.name.split(':')[0]: arg for arg in args}

      raw_features = {}
      for key in receiver_tensors:
        raw_features[key] = self._convert_to_sparse(receiver_tensors[key])

      transformed_features = self.tft_layer(raw_features)
      for key in transformed_features:
        transformed_features[key] = tf.expand_dims(
          transformed_features[key], axis=-1)
      logits = self(transformed_features, training=False)

      # Convert the label indices to strings and return the prediction
      _, pred_index = tf.math.top_k(logits, k=1)
      pred_index = tf.squeeze(pred_index, axis=-1)

      pred = tf.gather(all_labels, pred_index),
      return pred

    return serve_tf_examples_fn

  def get_config(self):
    return {
      'hidden_layer_dims': self.hidden_layer_dims,
      'num_labels': self.num_labels,
      'dropout_rate': self.dropout_rate
    }

  def _convert_to_sparse(self, a):
    batch_size = tf.shape(a)[0]
    indices = tf.cast(
      tf.expand_dims(tf.range(batch_size), -1),
      tf.int64
    )
    zeros = tf.zeros_like(indices)
    indices = tf.concat([indices, zeros], axis=1)
    sparse = tf.SparseTensor(
      indices=indices, values=a, dense_shape=[batch_size, 1]
    )

    return sparse
