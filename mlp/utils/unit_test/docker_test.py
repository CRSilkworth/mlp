from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import mlp.utils.docker as cur


class createUpdateRunTest(tf.test.TestCase):

  def test_remote_image_exists(self):
    image_name = 'tensorflow/tensorflow'
    image_tag = '2.8.0'

    result = cur.remote_image_exists(image_name, image_tag)
    self.assertTrue(result is True)

    image_name = 'tensorflow/tensorflow'
    image_tag = '2888888888888.8.0'

    result = cur.remote_image_exists(image_name, image_tag)
    self.assertTrue(result is False)

  def test_remote_image_exists(self):
    image_name = 'hello-world'
    image_tag = 'latest'

    cur.pull_image(image_name, image_tag)
    result = cur.local_image_exists(image_name, image_tag)
    self.assertTrue(result is True)

    image_name = 'hello-world'
    image_tag = '0.333333.3'

    result = cur.local_image_exists(image_name, image_tag)
    self.assertTrue(result is False)

if __name__ == '__main__':
  tf.test.main()
