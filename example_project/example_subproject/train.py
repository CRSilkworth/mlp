"""Trainer defintion."""
from __future__ import division
from __future__ import print_function
from typing import Optional, Dict, List, Text, Any, Callable

import absl
import os
import tensorflow as tf
import tensorflow_transform as tft

from tensorflow_metadata.proto.v0 import schema_pb2

from __example_subproject__ import model
from __example_subproject__ import preprocess as pre

from mlp.tensorflow.learning_rate_schedules import piecewise_learning_rate
from mlp.tensorflow.loggers import NBatchLogger


class Obj(dict):
  __getattr__ = dict.get


def trainer_factory(
  categorical_feature_keys,
  numerical_feature_keys,
  label_key: Optional[Text] = 'category'
) -> Callable:
  """
  Define a run_fn function to pass to the Trainer component.

  Parameters
  ----------
  label_key: the data key of the label.

  Returns
  -------
  run_fn: The defined run_fn

  """
  # def run_fn(
  def run_fn(
    hparams: Any,
    ) -> Dict[Text, Any]:
    """Build the estimator using the high level API.

    Parameters
    ----------
      hparams: Holds hyperparameters used to train the model as name/value pairs.

    Returns
    -------
      Dict:
        - estimator: The estimator that will be used for training and eval.
        - train_spec: Spec for training.
        - eval_spec: Spec for eval.
        - eval_input_receiver_fn: Input function for eval.

    """
    # Pull in the transform definition
    tf_transform_output = tft.TFTransformOutput(hparams.transform_output)

    custom_config = Obj(hparams.custom_config)

    # Build the inputs for the training and eval
    train_dataset = pre.get_input_fn(
      file_names=hparams.train_files,
      tf_transform_output=tf_transform_output,
      batch_size=custom_config.batch_size,
      label_key=custom_config.label_key,
    )
    eval_dataset = pre.get_input_fn(
      file_names=hparams.eval_files,
      tf_transform_output=tf_transform_output,
      batch_size=custom_config.batch_size,
      label_key=custom_config.label_key,
    )

    strategy = choose_strategy(custom_config.num_gpus)

    with strategy.scope():
      # Define the full model to be used in prediction.
      # Pull out all valid labels and their count
      vocabularies = {}
      for key in categorical_feature_keys + [label_key]:
        vocabularies[key] = tf_transform_output.vocabulary_by_name(key)

      num_labels = len(vocabularies[label_key])

      # Define the full model to be used in prediction.
      basic_nn = model.BasicNN(
        hidden_layer_dims=custom_config.hidden_layer_dims,
        num_labels=num_labels,
        vocabularies=vocabularies,
        tf_transform_output_dir=hparams.transform_output
      )

      if hasattr(hparams, 'base_model') and hparams.base_model is not None:
        latest_checkpoint = tf.train.latest_checkpoint(
          os.path.join(hparams.base_model, 'checkpoints'))
        absl.logging.info("Loading weights from {}".format(latest_checkpoint))
        basic_nn.load_weights(latest_checkpoint)

      ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
          hparams.serving_model_dir, 'checkpoints', 'ckpt_{epoch}'
        ),
        verbose=1
      )
      print_every = NBatchLogger(custom_config.save_summary_steps)

      optimizer = tf.keras.optimizers.Adam(
        learning_rate=custom_config.learning_rate)

      loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

      basic_nn.compile(
        optimizer=optimizer,
        loss=loss,
      )

      basic_nn.fit(
        train_dataset,
        epochs=custom_config.num_epochs,
        steps_per_epoch=custom_config.steps_per_epoch,
        validation_data=eval_dataset,
        validation_steps=custom_config.num_eval_steps,
        callbacks=[print_every, ckpt_callback],
        verbose=2
      )

      signatures = {
        'serving_default': basic_nn.get_serve_tf_examples_fn(
          categorical_feature_keys, numerical_feature_keys, vocabularies[label_key])
      }

      basic_nn.save(hparams.serving_model_dir, save_format='tf', signatures=signatures)

  return run_fn


def choose_strategy(num_gpus):
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
