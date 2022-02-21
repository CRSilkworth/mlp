from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from mlp.data_generation.number import *
import numpy as np

class numberTest(tf.test.TestCase):

  def testFraction(self):
    pattern = np.array(['fraction']*6)
    language = np.array(['en', 'en', 'en', 'zh_hans', 'zh_hans', 'zh_hans'])
    whole = np.array([True]*6)
    number = np.array([
      0.5,
      .25,
      1./3.,
      25./4.,
      675/23.,
      .000021
    ], dtype=np.float32)
    string = np.array(['1/2', '1/4', '1/3', '6 1/4', '29 173913/500000', '21/1000000'])
    self.to_and_from_test(number, string, pattern, language=language, whole=whole)

  def to_and_from_test(self, number, string, pattern, **kwargs):
    string_out = number_to_string(number, pattern, **kwargs)
    self.assertAllEqual(string_out, string)

    number_out = string_to_number(string, pattern, **kwargs)
    self.assertAllEqual(number_out, number)

if __name__ == '__main__':
  tf.test.main()
