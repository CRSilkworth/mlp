"""ArtifactPusher component definition."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, Optional, Text, Union, List, Type

import absl
import os
import tensorflow as tf

from tfx import types
from tfx.dsl.components.base import base_component
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ComponentSpec
from tfx.types.component_spec import ExecutionParameter
from tfx.dsl.components.base import executor_spec
from tfx.proto import pusher_pb2
from tfx.types import standard_artifacts
from tfx.components.pusher.executor import Executor
from google.protobuf import json_format
from tfx.types import artifact_utils
from tfx.utils import io_utils

ARTIFACT_KEY = 'artifact'
PUSHED_ARTIFACT_KEY = 'pushed_artifact'

def copy_dir(src: Text, dst: Text) -> None:
  """Copies the whole directory recursively from source to destination."""

  if tf.io.gfile.exists(dst):
    tf.io.gfile.rmtree(dst)
  tf.io.gfile.makedirs(dst)

  for dir_name, sub_dirs, leaf_files in tf.io.gfile.walk(src):
    for leaf_file in leaf_files:
      leaf_file_path = os.path.join(dir_name, leaf_file)
      new_file_path = os.path.join(dir_name.replace(src, dst, 1), leaf_file)
      tf.io.gfile.copy(leaf_file_path, new_file_path)

    for sub_dir in sub_dirs:
      tf.io.gfile.makedirs(os.path.join(dir_name.replace(src, dst, 1), sub_dir))

class ArtifactPusherExecutor(Executor):
  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Push model to target directory if blessed.

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - model_export: exported model from trainer.
        - model_blessing: model blessing path from model_validator.  A push
          action delivers the model exports produced by Trainer to the
          destination defined in component config.
      output_dict: Output dict from key to a list of artifacts, including:
        - model_push: A list of 'ModelPushPath' artifact of size one. It will
          include the model in this push execution if the model was pushed.
      exec_properties: A dict of execution properties, including:
        - push_destination: JSON string of pusher_pb2.PushDestination instance,
          providing instruction of destination to push artifact.

    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)
    artifact_export = artifact_utils.get_single_instance(input_dict[ARTIFACT_KEY])
    artifact_path = artifact_export.uri

    artifact_push = artifact_utils.get_single_instance(
      output_dict[PUSHED_ARTIFACT_KEY]
    )

    push_destination = pusher_pb2.PushDestination()
    json_format.Parse(exec_properties['push_destination'], push_destination)

    destination_kind = push_destination.WhichOneof('destination')
    if destination_kind == 'filesystem':
      fs_config = push_destination.filesystem
      serving_path = fs_config.base_directory

      copy_dir(artifact_path, serving_path)
      absl.logging.info('artifact written to serving path %s.', serving_path)
    else:
      raise NotImplementedError(
          'Invalid push destination {}'.format(destination_kind))

    # Copy the artifact to pushing uri for archiving.
    copy_dir(artifact_path, artifact_push.uri)
    absl.logging.info('artifact pushed to %s.', artifact_push.uri)


def pusher_component_factory(artifact_type):
  class ArtifactPusherSpec(ComponentSpec):
    PARAMETERS = {
      'push_destination': ExecutionParameter(type=pusher_pb2.PushDestination, optional=True),
    }
    INPUTS = {
      'artifact': ChannelParameter(type=artifact_type)
    }
    OUTPUTS = {
      'pushed_artifact': ChannelParameter(type=artifact_type),
    }

  class ArtifactPusher(base_component.BaseComponent):
    SPEC_CLASS = ArtifactPusherSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(ArtifactPusherExecutor)

    def __init__(
        self,
        artifact: types.Channel,
        push_destination: Union[pusher_pb2.PushDestination, Dict[Text, Any]],
        properties: Optional[Dict[Text, Union[Text, int]]] = None,
        enable_cache: Optional[bool] = None
      ):
      """Construct a Pusher component.

      Args:
        artifact: A Channel of type `standard_artifacts.Artifact`, usually produced by
          a ArtifactGen component.
        push_destination: A pusher_pb2.PushDestination instance. Optional if executor_class
          doesn't require push_destination. If any field is provided as a
          RuntimeParameter, push_destination should be constructed as a dict with
          the same field names as PushDestination proto message.
        output: Optional output `standard_artifacts.PushedModel` channel with
          result of push.
        instance_name: Optional unique instance name. Necessary if multiple Pusher
          components are declared in the same pipeline.
        enable_cache: Optional boolean to indicate if cache is enabled for the
          Pusher component. If not specified, defaults to the value
          specified for pipeline's enable_cache parameter.
      """
      self._properties = properties or {}
      output = artifact_type()
      for key, value in self._properties.items():
        setattr(output, key, value)
      output = types.Channel(type=artifact_type, artifacts=[output])

      spec = ArtifactPusherSpec(
        artifact=artifact,
        push_destination=push_destination,
        pushed_artifact=output
      )
      super(ArtifactPusher, self).__init__(
        spec=spec)
  return ArtifactPusher


SchemaPusher = pusher_component_factory(standard_artifacts.Schema)
ModelPusher = pusher_component_factory(standard_artifacts.Model)
ModelBlessingPusher = pusher_component_factory(standard_artifacts.ModelBlessing)
TransformGraphPusher = pusher_component_factory(standard_artifacts.TransformGraph)
ExamplesPusher = pusher_component_factory(standard_artifacts.Examples)
