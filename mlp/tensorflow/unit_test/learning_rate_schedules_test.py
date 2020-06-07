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
"""Tests for slack component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import mlp.common.tensorflow.learning_rate_schedules as lrs
from mlp.common.proto import mysql_tunnel_config_pb2

class LearningRateScheduleTest(tf.test.TestCase):

  def setUp(self):
    super(LearningRateScheduleTest, self).setUp()

  def testPiecewiseLearningRate(self):
    learning_rate = 100.0
    num_train_steps = 100
    num_warmup_steps = 10
    num_cool_down_steps = 10

    # adj_learning_rate = lrs.piecewise_learning_rate(global_step, learning_rate, num_train_steps, num_warmup_steps, num_cool_down_steps)
    @tf.function
    def train_step(global_step):
      adj_learning_rate = lrs.piecewise_learning_rate(global_step, learning_rate, num_train_steps, num_warmup_steps, num_cool_down_steps)

      return adj_learning_rate

    global_steps = tf.cast(list(range(num_train_steps)), tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices(
      global_steps
    )

    for gs in dataset.take(num_train_steps):
      train_step(gs).numpy()

if __name__ == '__main__':
  tf.test.main()
