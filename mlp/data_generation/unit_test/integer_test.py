from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from mlp.data_generation.integer import *
import numpy as np

class IntegerTest(tf.test.TestCase):

  def testStringToInteger(self):
    pattern = np.array(['cast', 'comma', 'cast', 'comma'])
    integer = np.array([1, 100000, 2, 29999999])
    string = np.array(['1', '100,000', '2', '29,999,999'])
    self.to_and_from_test(integer, string, pattern)

  def testComma(self):
    pattern = np.array(['comma'] * 3)
    integer = np.array([100000, 29999999, 3888888888])
    string = np.array(['100,000', '29,999,999', '3,888,888,888'])
    self.to_and_from_test(integer, string, pattern)

  def testWord(self):
    pattern = np.array(['word']*6)
    language = np.array(['en'] * 3 + ['zh_hant'] * 3)
    integer = np.array([
      23478902,
      2362370945,
      789234875895,
      23478902,
      2362370945,
      789234875895,
    ])
    string = np.array([
      'twenty three million four hundred seventy eight thousand nine hundred two',
      'two billion three hundred sixty two million three hundred seventy thousand nine hundred forty five',
      'seven hundred eighty nine billion two hundred thirty four million eight hundred seventy five thousand eight hundred ninety five',
      '二千三百四十七万八千九百二', '二十三億六千二百三十七万九百四十五', '七千八百九十二億三千四百八十七万五千八百九十五'
      ])

    self.to_and_from_test(integer, string, pattern, language=language)

  def testCapWord(self):
    pattern = np.array(['cap_word']*3)
    language = np.array(['en']*3)
    integer = np.array([
      23478902,
      2362370945,
      789234875895,
    ])
    string = np.array([
      'Twenty three million four hundred seventy eight thousand nine hundred two',
      'Two billion three hundred sixty two million three hundred seventy thousand nine hundred forty five',
      'Seven hundred eighty nine billion two hundred thirty four million eight hundred seventy five thousand eight hundred ninety five'])

    self.to_and_from_test(integer, string, pattern, language=language)

  def testFloatWord(self):
    pattern = np.array(['float_word', 'float_word', 'float_word', 'float_word', 'float_word', 'float_word'])
    language = np.array(['en', 'en', 'en', 'zh_hans', 'zh_hans', 'zh_hans'])
    integer = np.array([
      1200,
      2500000,
      100200000000,
      1200,
      2500000,
      100200000000,
    ])
    string = np.array(['1.2 thousand', '2.5 million', '100.2 billion', '1.2千', '250万', '1002億'])
    self.to_and_from_test(integer, string, pattern, language=language)

  def to_and_from_test(self, integer, string, pattern, **kwargs):
    string_out = integer_to_string(integer, pattern, **kwargs)
    self.assertAllEqual(string_out, string)

    integer_out = string_to_integer(string, pattern, **kwargs)
    self.assertAllEqual(integer_out, integer)

if __name__ == '__main__':
  tf.test.main()
