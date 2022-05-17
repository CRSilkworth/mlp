from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from official.nlp import bert_modeling as modeling
from tensorflow_metadata.proto.v0 import schema_pb2
from tfx.components.trainer import executor as trainer_executor
from __example_subproject__ import model


class ModelTest(tf.test.TestCase):
  def testModel(self):
    pass


if __name__ == '__main__':
  tf.test.main()
