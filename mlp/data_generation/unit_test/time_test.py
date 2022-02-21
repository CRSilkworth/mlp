from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from mlp.data_generation.time import *
import numpy as np


class TimeTest(tf.test.TestCase):
  def to_and_from_test(self, string, hour, minute, second, pattern, **kwargs):
    string_out = time_to_string(hour, minute, second, pattern, **kwargs)
    self.assertAllEqual(string_out, string)

    hour_out, minute_out, second_out = string_to_time(string, pattern, **kwargs)
    self.assertAllEqual(hour_out, hour)
    self.assertAllEqual(minute_out, minute)
    self.assertAllEqual(second_out, second)

  def test_digit_hour(self):
    pattern = np.array(['digit_hour'] * 4)
    language = np.array(['en', 'en', 'ja', 'ko'])
    digit_type = np.array(['word', 'digit', 'full_width', 'digit'])
    am_pm_type = np.array([' am', ' at night', 'お昼の', '오전'])
    hour_word = np.array(['', 'oclock', '時', ''])
    zeros = np.array([False, False, True, False])

    hour = np.array([3, 17, 16, 11])
    minute = np.array([0, 0, 0, 0])
    second = np.array([0, 0, 0, 0])

    string = np.array([
      'three am',
      '5oclock at night',
      'お昼の０４時',
      '오전11'
    ])

    self.to_and_from_test(string, hour, minute, second, pattern, language=language, digit_type=digit_type, am_pm_type=am_pm_type, hour_word=hour_word, zeros=zeros)

  def test_digit_hour_minute(self):
    pattern = np.array(['digit_hour_minute'] * 4)
    language = np.array(['en', 'en', 'ja', 'ko'])
    digit_type = np.array(['word', 'digit', 'full_width', 'digit'])
    am_pm_type = np.array([' am', ' at night', 'お昼の', '오전'])
    hour_word = np.array(['', ':', '時', '시'])
    minute_word = np.array(['', '', '', '분'])
    short_cut_word = np.array(['quarter past ', '', '半', ''])
    zeros = np.array([False, False, False, False])

    hour = np.array([3, 17, 16, 11])
    minute = np.array([15, 20, 30, 2])
    second = np.array([0, 0, 0, 0])

    string = np.array([
      'quarter past three am',
      '5:20 at night',
      'お昼の４時半',
      '오전11시2분'
    ])

    self.to_and_from_test(string, hour, minute, second, pattern, language=language, digit_type=digit_type, am_pm_type=am_pm_type, hour_word=hour_word, zeros=zeros, minute_word=minute_word, short_cut_word=short_cut_word)

if __name__ == '__main__':
  tf.test.main()
