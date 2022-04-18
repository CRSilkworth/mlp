# Lint as: python2, python3
# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TFX Pusher component definition."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from typing import Any, Dict, Optional, Text, Union, List

from absl import logging

from google.protobuf import json_format
from tfx import types
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_executor
from tfx.dsl.components.base import executor_spec
from tfx.utils import io_utils
from mlp.utils import dirs
from tfx.utils import path_utils
from tfx.proto import pusher_pb2
from tfx.types import standard_artifacts
from tfx.types import artifact_utils
from tfx.types.standard_component_specs import PusherSpec
from tfx.types.component_spec import ExecutionParameter
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ComponentSpec
from tfx.proto import pusher_pb2
import time


# Key for model in executor input_dict.
MODEL_KEY = 'model'

# Key for pushed model in executor output_dict.
PUSHED_MODEL_KEY = 'pushed_model'


class AlwaysPusherExecutor(base_executor.BaseExecutor):
  """TFX Pusher executor to push the new TF model to a filesystem target."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Push model to target directory if blessed.

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - model_export: exported model from trainer.
      output_dict: Output dict from key to a list of artifacts, including:
        - model_push: A list of 'ModelPushPath' artifact of size one. It will
          include the model in this push execution if the model was pushed.
      exec_properties: A dict of execution properties, including:
        - push_destination: JSON string of pusher_pb2.PushDestination instance,
          providing instruction of destination to push model.

    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)
    model_push = artifact_utils.get_single_instance(
        output_dict[PUSHED_MODEL_KEY])

    model_push_uri = model_push.uri
    model_export = artifact_utils.get_single_instance(input_dict[MODEL_KEY])
    model_export_uri = model_export.uri
    logging.info('Model pushing.')
    # Copy the model to pushing uri.
    model_path = path_utils.serving_model_path(model_export_uri)
    model_version = str(int(time.time()))
    # model_version = path_utils.get_serving_model_version(model_export_uri)
    logging.info('Model version is %s', model_version)
    io_utils.copy_dir(model_path, os.path.join(model_push_uri, model_version))
    logging.info('Model written to %s.', model_push_uri)

    push_destination = pusher_pb2.PushDestination()
    json_format.Parse(exec_properties['push_destination'], push_destination)
    serving_path = os.path.join(push_destination.filesystem.base_directory,
                                model_version)
    if tf.io.gfile.exists(serving_path):
      logging.info(
          'Destination directory %s already exists, skipping current push.',
          serving_path)
    else:
      # tf.serving won't load partial model, it will retry until fully copied.
      # io_utils.copy_dir(model_path, serving_path)
      dirs.copy_dir(model_path, serving_path, ignore_subdirs=['checkpoints'])
      logging.info('Model written to serving path %s.', serving_path)

    model_push.set_int_custom_property('pushed', 1)
    model_push.set_string_custom_property('pushed_model', model_export_uri)
    model_push.set_int_custom_property('pushed_model_id', model_export.id)
    logging.info('Model pushed to %s.', serving_path)


class AlwaysPusherSpec(ComponentSpec):
  """Pusher component spec."""

  PARAMETERS = {
    'push_destination':
        ExecutionParameter(type=pusher_pb2.PushDestination, optional=True),
    'custom_config':
        ExecutionParameter(type=Dict[Text, Any], optional=True),
  }
  INPUTS = {
    'model': ChannelParameter(type=standard_artifacts.Model),
  }
  OUTPUTS = {
    'pushed_model': ChannelParameter(type=standard_artifacts.PushedModel),
  }


class AlwaysPusher(base_component.BaseComponent):
  SPEC_CLASS = AlwaysPusherSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(AlwaysPusherExecutor)

  def __init__(
      self,
      model: types.Channel = None,
      push_destination: Optional[Union[pusher_pb2.PushDestination,
                                       Dict[Text, Any]]] = None,
      custom_config: Optional[Dict[Text, Any]] = None,
      custom_executor_spec: Optional[executor_spec.ExecutorSpec] = None,
      output: Optional[types.Channel] = None,
    ):
    """Construct a Pusher component.

    Args:
      model: A Channel of type `standard_artifacts.Model`, usually produced by
        a Trainer component.
      push_destination: A pusher_pb2.PushDestination instance, providing info
        for tensorflow serving to load models. Optional if executor_class
        doesn't require push_destination. If any field is provided as a
        RuntimeParameter, push_destination should be constructed as a dict with
        the same field names as PushDestination proto message.
      custom_config: A dict which contains the deployment job parameters to be
        passed to cloud-based training platforms.  The [Kubeflow
          example](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_kubeflow.py#L211)
            contains an example how this can be used by custom executors.
      custom_executor_spec: Optional custom executor spec.
      output: Optional output `standard_artifacts.PushedModel` channel with
        result of push.
      instance_name: Optional unique instance name. Necessary if multiple Pusher
        components are declared in the same pipeline.
    """
    output = output or types.Channel(
        type=standard_artifacts.PushedModel,
        artifacts=[standard_artifacts.PushedModel()])

    if push_destination is None and not custom_executor_spec:
      raise ValueError('push_destination is required unless a '
                       'custom_executor_spec is supplied that does not require '
                       'it.')
    spec = AlwaysPusherSpec(
        model=model,
        push_destination=push_destination,
        custom_config=custom_config,
        pushed_model=output)

    super(AlwaysPusher, self).__init__(
        spec=spec,
        custom_executor_spec=custom_executor_spec,
      )
