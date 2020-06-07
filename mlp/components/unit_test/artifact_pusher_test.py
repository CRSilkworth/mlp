"""Tests for pandas_window_example_gen component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

from mlp.components import artifact_pusher as component
from google.protobuf import json_format

from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.proto import pusher_pb2

class ComponentTest(tf.test.TestCase):
  def testConstructSchema(self):
    output_data_dir = os.path.join(
      os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()), self._testMethodName)
    artifact_channel = channel_utils.as_channel([standard_artifacts.Schema()])
    component_instance = component.SchemaPusher(
      artifact=artifact_channel,
      push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
          base_directory=output_data_dir
        )
      )
    )
    self.assertEqual(
      'Schema',
      component_instance.inputs.artifact.type_name
    )
    self.assertEqual(
      'Schema',
      component_instance.outputs.pushed_artifact.type_name
    )

  def testConstructModel(self):
    output_data_dir = os.path.join(
      os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()), self._testMethodName)
    artifact_channel = channel_utils.as_channel([standard_artifacts.Model()])
    component_instance = component.ModelPusher(
      artifact=artifact_channel,
      push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
          base_directory=output_data_dir
        )
      )
    )
    self.assertEqual(
      'Model',
      component_instance.inputs.artifact.type_name
    )
    self.assertEqual(
      'Model',
      component_instance.outputs.pushed_artifact.type_name
    )

  def testConstructTransformGraph(self):
    output_data_dir = os.path.join(
      os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()), self._testMethodName)
    artifact_channel = channel_utils.as_channel([standard_artifacts.TransformGraph()])
    component_instance = component.TransformGraphPusher(
      artifact=artifact_channel,
      push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
          base_directory=output_data_dir
        )
      )
    )
    self.assertEqual(
      'TransformGraph',
      component_instance.inputs.artifact.type_name
    )
    self.assertEqual(
      'TransformGraph',
      component_instance.outputs.pushed_artifact.type_name
    )

  def testConstructExamples(self):
    output_data_dir = os.path.join(
      os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()), self._testMethodName)
    artifact_channel = channel_utils.as_channel([standard_artifacts.Examples()])
    component_instance = component.ExamplesPusher(
      artifact=artifact_channel,
      push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
          base_directory=output_data_dir
        )
      )
    )
    self.assertEqual(
      'Examples',
      component_instance.inputs.artifact.type_name
    )
    self.assertEqual(
      'Examples',
      component_instance.outputs.pushed_artifact.type_name
    )

  def testDo(self):
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    tf.io.gfile.makedirs(os.path.join(output_data_dir, 'input'))
    tf.io.gfile.makedirs(os.path.join(output_data_dir, 'output'))

    input = standard_artifacts.ModelBlessing()
    input.uri = os.path.join(output_data_dir, 'input')
    with open(os.path.join(output_data_dir, 'input', 'BLESSED'), 'w') as f:
      f.write('')

    output = standard_artifacts.ModelBlessing()
    output.uri = os.path.join(output_data_dir, 'output')
    input_dict = {'artifact': [input]}
    output_dict = {'pushed_artifact': [output]}
    exec_properties = {
      'push_destination':
      json_format.MessageToJson(pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
          base_directory=output_data_dir
        )
      ),
      sort_keys=True
      )
    }
    # Run executor.
    executor = component.ArtifactPusherExecutor()
    executor.Do(input_dict, output_dict, exec_properties)

if __name__ == '__main__':
  tf.test.main()
