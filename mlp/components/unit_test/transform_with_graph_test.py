"""Tests for pandas_window_example_gen component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

from mlp.components import transform_with_graph as component
from google.protobuf import json_format

from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.proto import pusher_pb2


class ComponentTest(tf.test.TestCase):
  def testConstruct(self):
    output_data_dir = os.path.join(
      os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()), self._testMethodName)
    examples_channel = channel_utils.as_channel([standard_artifacts.Examples()])
    schema_channel = channel_utils.as_channel([standard_artifacts.Schema()])
    transform_graph_channel = channel_utils.as_channel([standard_artifacts.TransformGraph()])
    component_instance = component.TransformWithGraph(
      examples=examples_channel,
      schema=schema_channel,
      transform_graph=transform_graph_channel,
    )
    self.assertEqual(
      'Schema',
      component_instance.inputs.schema.type_name
    )
    self.assertEqual(
      'Examples',
      component_instance.inputs.examples.type_name
    )
    self.assertEqual(
      'TransformGraph',
      component_instance.inputs.transform_graph.type_name
    )
    self.assertEqual(
      'Examples',
      component_instance.outputs.transformed_examples.type_name
    )


if __name__ == '__main__':
  tf.test.main()
