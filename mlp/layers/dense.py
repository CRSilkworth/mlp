import tensorflow as tf
from typing import Optional, List, Text, Dict


class IndexedDense(tf.keras.layers.Layer):
  """Creates a dense layer which has different weights and biases which are chosen according to the index tensor.

  The total shape of the weights tensor is (num_indices, input_dim, units) while
  the total shape of the bias tensor is (num_indices, units).

  The shape of the input, i.e. the tensor to be transformed, is (batch_size, input_dim) while the shape of the index is (batch_size,). The index tensor takes slices from the weights and bias tensors does a regular affine transformation to the input tensor. Afterwards an activation layer is performed.
  """
  def __init__(
    self,
    units: int,
    num_indices: int,
    activation: Optional[str] = None,
    use_bias: Optional[bool] = True,
    kernel_initializer: Optional[str] = 'glorot_uniform',
    bias_initializer: Optional[str] = 'zeros',
    dtype: Optional[tf.DType] = tf.float32,
    **kwargs
    ):
    """Define the index layer.

    Parameters
    ----------
      units: The dimension of the output.
      num_indices: The number of different weights and bias slices to create.
      activation: Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the kernel weights matrix.
      bias_initializer: Initializer for the bias vector.
      dtype: The data type of the output tensor.
    """
    super(IndexedDense, self).__init__(**kwargs)
    self.units = units
    self.num_indices = num_indices
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.num_indices = num_indices
    self.activation = activation
    # self.dtype = dtype

    self.activation_layer = tf.keras.layers.Activation(activation, dtype=dtype)

  def build(self, input_shape):
    self.ws = self.add_weight(
      shape=(self.num_indices, input_shape[0][-1], self.units),
      initializer=self.kernel_initializer,
      trainable=self.trainable
    )
    self.bs = self.add_weight(
      shape=(self.num_indices, self.units),
      initializer=self.bias_initializer,
      trainable=self.trainable
    )

  def call(self, inputs):
    input, index = inputs

    # Pull out the weight and bias slices from the full tensors.
    weights = tf.gather_nd(
      self.ws,
      index
    )
    bias = tf.gather_nd(
      self.bs,
      index
    )

    # Handles the case where the input has more than just a batch dimension and
    # input dimension. e.g. when there is a time dimension. Tiles the bias so
    # that it adds the same bias to each time step.
    if len(input.shape) > 2:
      for dim in input.shape[1:-1]:
        bias = tf.expand_dims(bias, axis=1)

      shape = input.shape
      bias = tf.tile(
        bias,
        multiples=[1] + shape[1:-1] + [1]
      )

    output = tf.einsum('i...j,i...jk->i...k', input, weights) + bias
    output = self.activation_layer(output)
    return output
