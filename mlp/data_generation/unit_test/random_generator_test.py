from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import mlp.data_generation.random_generator as rg
import mlp.data_generation.maps as maps_dg
import numpy as np
import pandas as pd

class TimeTest(tf.test.TestCase):
  def setUp(self):
    self.random_value_parameters = {
      0: {
        'categorical': {
          'pattern': ['cast', 'comma'],
          'language': ['en', 'ja', 'zh_hans', 'zh_hant', 'ko']
          },
        'bounded': {
        },
        'normal': {
        },
        'integer': {
          'integer': 2.0
        },
      },
      1: {
        'categorical': {
          'pattern': ['word', 'float_word'],
          'language': ['en', 'ja', 'zh_hans', 'zh_hant', 'ko']
        },
        'bounded': {
        },
        'normal': {
        },
        'integer': {
          'integer': 2.0
        },
      },
      2: {
        'categorical': {
          'pattern': ['delta_day_of_week', 'delta_day', 'd_word_y'],
          'language': ['en', 'ja', 'zh_hans', 'zh_hant', 'ko'],
          'cap': [True, False],
          'abbr': [True, False],
          'era': [True, False],
          'no_year': [True, False],
          'zeros': [True, False],
          'delimiter': ['-', '/', '_', ' ', ''],
          'postfix': [True, False],
          'next': ['this', 'next', 'last'],
          'num_days': [-2, -1, 0, 1, 2]

        },

        'bounded': {
          'num': 2
          },
        'normal': {
          'delta_day': [18993, 5479],
          'cur_delta_day': [18993, 5479]

        },
      },
      3: {
        'categorical': {
          'pattern': ['digit_hour', 'digit_hour_minute'],
          'language': ['en', 'ja', 'zh_hans', 'zh_hant', 'ko'],
          'digit_type': ['word', 'alt_word_0', 'full_width', 'digit'],
          'am_pm_type': maps_dg.all_am_pm_types,
          'zeros': [True, False],
          'hour_word': maps_dg.all_hour_words,
          'minute_word': maps_dg.all_minute_words,
          'short_cut_word': maps_dg.all_short_cut_words,
        },
        'bounded': {
          'hour': 24,
          'minute': 60,
          'second': 1
          },
        'normal': {
        },
        'integer': {
        },
      },
    }

    self.random_seed = 0

  def test_init(self):
    rdg = rg.RandomDataGenerator(self.random_value_parameters)

  def test_get_random_draw(self):
    rdg = rg.RandomDataGenerator(self.random_value_parameters, random_seed=self.random_seed)

    random_draw = rdg.get_random_draw(self.random_value_parameters[0], 4)

    true_random_draw = {
      'pattern': np.array(['cast', 'comma', 'comma', 'cast']),
      'language': np.array(['zh_hant', 'zh_hant', 'zh_hant', 'ja']),
      'integer': np.array([3007, 0, 64, 9]),
    }

    for key in true_random_draw:
      try:
        self.assertAllEqual(random_draw[key], true_random_draw[key])
      except Exception as e:
        print("{} failed".format(key))
        raise e

  def test_float(self):
    rdg = rg.RandomDataGenerator(self.random_value_parameters, random_seed=self.random_seed)

    df = pd.DataFrame({'data_type_id': [1, 1, 1, 1, 1]})

    df = rdg._generate_float(df, self.random_value_parameters[0])
    true_df = pd.DataFrame({
      'span': ['1207', '0', '64', '9', '6,000,000,012'],
      'float': [1207., 0., 64., 9., 6000000012.]
    })

    for key in ['span', 'float']:
      self.assertAllEqual(df[key], true_df[key])

  def test_integer(self):
    rdg = rg.RandomDataGenerator(self.random_value_parameters, random_seed=self.random_seed)

    df = pd.DataFrame({'data_type_id': [0, 0, 0, 0, 0]})

    df = rdg._generate_integer(df, self.random_value_parameters[0])
    true_df = pd.DataFrame({
      'span': ['1207', '0', '64', '9', '6,000,000,012'],
      'integer': [1207, 0, 64, 9, 6000000012]
    })

    for key in ['span', 'integer']:
      self.assertAllEqual(df[key], true_df[key])

  def test_date(self):
    rdg = rg.RandomDataGenerator(self.random_value_parameters, random_seed=self.random_seed)

    df = pd.DataFrame({'data_type_id': [2, 2, 2, 2, 2]})

    df = rdg._generate_date(df, self.random_value_parameters[2])
    true_df = pd.DataFrame({'data_type_id': {0: 2, 1: 2, 2: 2, 3: 2, 4: 2}, 'span': {0: '上个周六', 1: '그저께', 2: 'Last Wednesday', 3: 'day after tomorrow', 4: '그저께'}, 'year': {0: 2006, 1: 2031, 2: 2043, 3: 2018, 4: 2020}, 'month': {0: 2, 1: 3, 2: 6, 3: 10, 4: 10}, 'day': {0: 4, 1: 22, 2: 10, 3: 29, 4: 15}, 'cur_year': {0: 2006, 1: 2031, 2: 2043, 3: 2018, 4: 2020}, 'cur_month': {0: 2, 1: 3, 2: 6, 3: 10, 4: 10}, 'cur_day': {0: 6, 1: 24, 2: 13, 3: 27, 4: 17}})
    for key in ['span', 'year', 'month', 'day', 'cur_year', 'cur_month', 'cur_day']:
      self.assertAllEqual(df[key], true_df[key])

  def test_time(self):
    rdg = rg.RandomDataGenerator(self.random_value_parameters, random_seed=self.random_seed)

    df = pd.DataFrame({'data_type_id': [3, 3, 3, 3, 3]})

    df = rdg._generate_time(df, self.random_value_parameters[3])
    true_df = pd.DataFrame({'data_type_id': {0: 3, 1: 3, 2: 3, 3: 3, 4: 3}, 'span': {0: '下午的〇二', 1: '清晨07:00:', 2: '昼の０時３１', 3: '清晨的一', 4: '下午的九五二'}, 'hour': {0: 14, 1: 7, 2: 0, 3: 1, 4: 9}, 'minute': {0: 0, 1: 0, 2: 31, 3: 0, 4: 52}, 'second': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}})

    for key in ['span', 'hour', 'minute', 'second']:
      self.assertAllEqual(df[key], true_df[key])


if __name__ == '__main__':
  tf.test.main()
