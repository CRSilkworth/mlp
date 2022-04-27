from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from mlp.data_generation.date import *
import numpy as np

class DateTest(tf.test.TestCase):

  def testEraToYear(self):
    years = [1566, 1763, 2030, 1947, 1855]

    for year in years:
      num, era = year_to_era(year)
      self.assertAllEqual(era_to_year(num, era), year)

  def test_ymd(self):
    pattern = np.array(['ymd'] * 16)
    language = np.array(['en'] * 16)
    cur_year = np.array([2021, 1960, 2030, 1745] * 4)
    cur_month = np.array([3, 12, 1, 6] * 4)
    cur_day = np.array([27, 31, 1, 30] * 4)
    cur_day_of_week = np.array([5, 5, 1, 2] * 4)

    zeros = np.array([True] * 4 + [False] * 4 + [True] * 4 + [False] * 4)
    delimiter = np.array(['-'] * 8 + ['/'] * 8)
    no_year = np.array([False] * 16)

    year = np.array([2021, 1960, 2030, 1745] * 4)
    month = np.array([3, 12, 1, 6] * 4)
    day = np.array([27, 31, 1, 30] * 4)

    strings = []
    for d in ['-', '/']:
      string = np.array(['2021' + d + '03' + d + '27', '1960' + d + '12' + d + '31', '2030' + d + '01' + d + '01', '1745' + d + '06' + d + '30'])
      strings.append(string)

      string = np.array(['2021' + d + '3' + d + '27', '1960' + d + '12' + d + '31', '2030' + d + '1' + d + '1', '1745' + d + '6' + d + '30'])
      strings.append(string)

    string = np.concatenate(strings)

    self.to_and_from_test(string, year, month, day, pattern, language=language, cur_year=cur_year, cur_month=cur_month, cur_day=cur_day, cur_day_of_week=cur_day_of_week, zeros=zeros, delimiter=delimiter, no_year=no_year)

  def test_mdy(self):
    pattern = np.array(['mdy'] * 16)
    language = np.array(['en'] * 16)
    cur_year = np.array([2021, 1960, 2030, 1745] * 4)
    cur_month = np.array([3, 12, 1, 6] * 4)
    cur_day = np.array([27, 31, 1, 30] * 4)
    cur_day_of_week = np.array([5, 5, 1, 2] * 4)

    year = np.array([2021, 1960, 2030, 1745] * 4)
    month = np.array([3, 12, 1, 6] * 4)
    day = np.array([27, 31, 1, 30] * 4)

    delimiter = np.array(['-'] * 8 + ['/'] * 8)
    zeros = np.array([True] * 4 + [False] * 4 + [True] * 4 + [False] * 4)
    no_year = np.array([False] * 4 + [True] * 4 + [False] * 4 + [True] * 4)

    strings = []
    for d in ['-', '/']:
      string = np.array(['03' + d + '27' + d + '2021', '12' + d + '31' + d + '1960', '01' + d + '01' + d + '2030', '06' + d + '30' + d + '1745'])
      strings.append(string)
      # self.to_and_from_test(string, year, month, day, pattern, language=language, cur_year=cur_year, cur_month=cur_month, cur_day=cur_day, cur_day_of_week=cur_day_of_week, delimiter=delimiter)

      string = np.array(['3' + d + '27', '12' + d + '31', '1' + d + '1', '6' + d + '30'])
      strings.append(string)
      # self.to_and_from_test(string, year, month, day, pattern, language=language, cur_year=cur_year, cur_month=cur_month, cur_day=cur_day, cur_day_of_week=cur_day_of_week, zeros=False, delimiter=delimiter, no_year=True)
    string = np.concatenate(strings)
    self.to_and_from_test(string, year, month, day, pattern, language=language, cur_year=cur_year, cur_month=cur_month, cur_day=cur_day, cur_day_of_week=cur_day_of_week, zeros=zeros, delimiter=delimiter, no_year=no_year)

  def test_dmy(self):
    pattern = np.array(['dmy'] * 24)
    language = np.array(['en'] * 24)
    cur_year = np.array([2021, 1960, 2030, 1745] * 6)
    cur_month = np.array([3, 12, 1, 6] * 6)
    cur_day = np.array([27, 31, 1, 30] * 6)
    cur_day_of_week = np.array([5, 5, 1, 2] * 6)

    year = np.array([2021, 1960, 2030, 1745] * 6)
    month = np.array([3, 12, 1, 6] * 6)
    day = np.array([27, 31, 1, 30] * 6)

    delimiter = np.array(['-'] * 12 + ['/'] * 12)
    zeros = np.array([True] * 4 + [False] * 8 + [True] * 4 + [False] * 8)
    no_year = np.array([False] * 8 + [True] * 4 + [False] * 8 + [True] * 4)

    strings = []
    for d in ['-', '/']:
      string = np.array(['27' + d + '03' + d + '2021', '31' + d + '12' + d + '1960', '01' + d + '01' + d + '2030', '30' + d + '06' + d + '1745'])
      strings.append(string)

      string = np.array(['27' + d + '3' + d + '2021', '31' + d + '12' + d + '1960', '1' + d + '1' + d + '2030', '30' + d + '6' + d + '1745'])
      strings.append(string)

      string = np.array(['27' + d + '3', '31' + d + '12', '1' + d + '1', '30' + d + '6'])
      strings.append(string)

    string = np.concatenate(strings)

    self.to_and_from_test(string, year, month, day, pattern, language=language, cur_year=cur_year, cur_month=cur_month, cur_day=cur_day, cur_day_of_week=cur_day_of_week, zeros=zeros, delimiter=delimiter, no_year=no_year)

  def test_word_dy(self):
    pattern = np.array(['word_dy'] * 16)
    language = np.array(['en'] * 16)
    cur_year = np.array([2021, 1960, 2030, 1745] * 4)
    cur_month = np.array([3, 12, 1, 6] * 4)
    cur_day = np.array([27, 31, 1, 30] * 4)
    cur_day_of_week = np.array([5, 5, 1, 2] * 4)

    year = np.array([2021, 1960, 2030, 1745] * 4)
    month = np.array([3, 12, 1, 6] * 4)
    day = np.array([27, 31, 1, 30] * 4)

    zeros = np.array([False] * 4 + [True] * 4 + [False] * 8)
    postfix = np.array([False] * 8 + [True] * 8)
    no_year = np.array([False] * 12 + [True] * 4)

    string = np.array([
      'March 27, 2021',
      'December 31, 1960',
      'January 1, 2030',
      'June 30, 1745',
      'March 27, 2021',
      'December 31, 1960',
      'January 01, 2030',
      'June 30, 1745',
      'March 27th, 2021',
      'December 31st, 1960',
      'January 1st, 2030',
      'June 30th, 1745',
      'March 27th',
      'December 31st',
      'January 1st',
      'June 30th'
    ])

    self.to_and_from_test(string, year, month, day, pattern, language=language, cur_year=cur_year, cur_month=cur_month, cur_day=cur_day, cur_day_of_week=cur_day_of_week, zeros=zeros, no_year=no_year, postfix=postfix)

    # string = np.array([
    #   'March 27, 2021',
    #   'December 31, 1960',
    #   'January 01, 2030',
    #   'June 30, 1745'
    # ])
    #
    # self.to_and_from_test(string, year, month, day, pattern, language=language, cur_year=cur_year, cur_month=cur_month, cur_day=cur_day, cur_day_of_week=cur_day_of_week, zeros=True)
    #
    # string = np.array([
    # 'March 27th, 2021',
    # 'December 31st, 1960',
    # 'January 1st, 2030',
    # 'June 30th, 1745'
    # ])
    #
    # self.to_and_from_test(string, year, month, day, pattern, language=language, cur_year=cur_year, cur_month=cur_month, cur_day=cur_day, cur_day_of_week=cur_day_of_week, postfix=True)
    #
    # string = np.array([
    #   'March 27th',
    #   'December 31st',
    #   'January 1st',
    #   'June 30th'
    # ])
    #
    # self.to_and_from_test(string, year, month, day, pattern, language=language, cur_year=cur_year, cur_month=cur_month, cur_day=cur_day, cur_day_of_week=cur_day_of_week, postfix=True, no_year=True)

  def test_d_word_y(self):
    pattern = np.array(['d_word_y'] * 16)
    language = np.array(['en'] * 16)
    cur_year = np.array([2021, 1960, 2030, 1745] * 4)
    cur_month = np.array([3, 12, 1, 6] * 4)
    cur_day = np.array([27, 31, 1, 30] * 4)
    cur_day_of_week = np.array([5, 5, 1, 2] * 4)

    year = np.array([2021, 1960, 2030, 1745] * 4)
    month = np.array([3, 12, 1, 6] * 4)
    day = np.array([27, 31, 1, 30] * 4)

    zeros = np.array([False] * 4 + [True] * 4 + [False] * 8)
    postfix = np.array([False] * 8 + [True] * 8)
    no_year = np.array([False] * 12 + [True] * 4)

    string = np.array([
      '27 March, 2021',
      '31 December, 1960',
      '1 January, 2030',
      '30 June, 1745',
      '27 March, 2021',
      '31 December, 1960',
      '01 January, 2030',
      '30 June, 1745',
      '27th March, 2021',
      '31st December, 1960',
      '1st January, 2030',
      '30th June, 1745',
      '27th March',
      '31st December',
      '1st January',
      '30th June'
    ])

    self.to_and_from_test(string, year, month, day, pattern, language=language, cur_year=cur_year, cur_month=cur_month, cur_day=cur_day, cur_day_of_week=cur_day_of_week, zeros=zeros, no_year=no_year, postfix=postfix)

    # string = np.array([
    #   '27 March, 2021',
    #   '31 December, 1960',
    #   '01 January, 2030',
    #   '30 June, 1745'
    # ])
    #
    # self.to_and_from_test(string, year, month, day, pattern, language=language, cur_year=cur_year, cur_month=cur_month, cur_day=cur_day, cur_day_of_week=cur_day_of_week, zeros=True)
    #
    # string = np.array([
    #   '27th March, 2021',
    #   '31st December, 1960',
    #   '1st January, 2030',
    #   '30th June, 1745'
    # ])
    #
    # self.to_and_from_test(string, year, month, day, pattern, language=language, cur_year=cur_year, cur_month=cur_month, cur_day=cur_day, cur_day_of_week=cur_day_of_week, postfix=True)
    #
    # string = np.array([
    #   '27th March',
    #   '31st December',
    #   '1st January',
    #   '30th June'
    # ])
    #
    # self.to_and_from_test(string, year, month, day, pattern, language=language, cur_year=cur_year, cur_month=cur_month, cur_day=cur_day, cur_day_of_week=cur_day_of_week, postfix=True, no_year=True)

  def test_ja(self):
    pattern = np.array(['ja'] * 12)
    language = np.array(['ja'] * 12)
    cur_year = np.array([2021, 1960, 2030, 1745] * 3)
    cur_month = np.array([3, 12, 1, 6] * 3)
    cur_day = np.array([27, 31, 1, 30] * 3)
    cur_day_of_week = np.array([5, 5, 1, 2] * 3)

    year = np.array([2021, 1960, 2030, 1745] * 3)
    month = np.array([3, 12, 1, 6] * 3)
    day = np.array([27, 31, 1, 30] * 3)

    zeros = np.array([False] * 4 + [True] * 4 + [False] * 4)
    no_year = np.array([False] * 8 + [True] * 4)

    string = np.array([
      '2021年3月27日',
      '1960年12月31日',
      '2030年1月1日',
      '1745年6月30日',
      '2021年03月27日',
      '1960年12月31日',
      '2030年01月01日',
      '1745年06月30日',
      '3月27日',
      '12月31日',
      '1月1日',
      '6月30日'
    ])

    self.to_and_from_test(string, year, month, day, pattern, language=language, cur_year=cur_year, cur_month=cur_month, cur_day=cur_day, cur_day_of_week=cur_day_of_week, zeros=zeros, no_year=no_year)

    # string = np.array([
    #   '2021年03月27日',
    #   '1960年12月31日',
    #   '2030年01月01日',
    #   '1745年06月30日'
    # ])
    #
    # self.to_and_from_test(string, year, month, day, pattern, language=language, cur_year=cur_year, cur_month=cur_month, cur_day=cur_day, cur_day_of_week=cur_day_of_week, zeros=True)
    #
    # string = np.array([
    #   '3月27日',
    #   '12月31日',
    #   '1月1日',
    #   '6月30日'
    # ])
    #
    # self.to_and_from_test(string, year, month, day, pattern, language=language, cur_year=cur_year, cur_month=cur_month, cur_day=cur_day, cur_day_of_week=cur_day_of_week, zeros=False, no_year=True)



  def test_ko(self):
    pattern = np.array(['ko'] * 12)
    language = np.array(['ko'] * 12)
    cur_year = np.array([2021, 1960, 2030, 1745] * 3)
    cur_month = np.array([3, 12, 1, 6] * 3)
    cur_day = np.array([27, 31, 1, 30] * 3)
    cur_day_of_week = np.array([5, 5, 1, 2] * 3)

    year = np.array([2021, 1960, 2030, 1745] * 3)
    month = np.array([3, 12, 1, 6] * 3)
    day = np.array([27, 31, 1, 30] * 3)

    zeros = np.array([False] * 4 + [True] * 4 + [False] * 4)
    no_year = np.array([False] * 8 + [True] * 4)

    string = np.array([
      '2021년3월27일',
      '1960년12월31일',
      '2030년1월1일',
      '1745년6월30일',
      '2021년03월27일',
      '1960년12월31일',
      '2030년01월01일',
      '1745년06월30일',
      '3월27일',
      '12월31일',
      '1월1일',
      '6월30일'
    ])

    self.to_and_from_test(string, year, month, day, pattern, language=language, cur_year=cur_year, cur_month=cur_month, cur_day=cur_day, cur_day_of_week=cur_day_of_week, zeros=zeros, no_year=no_year)

    # string = np.array([
    #   '2021년03월27일',
    #   '1960년12월31일',
    #   '2030년01월01일',
    #   '1745년06월30일'
    # ])
    #
    # self.to_and_from_test(string, year, month, day, pattern, language=language, cur_year=cur_year, cur_month=cur_month, cur_day=cur_day, cur_day_of_week=cur_day_of_week, zeros=True)
    #
    # string = np.array([
    #   '3월27일',
    #   '12월31일',
    #   '1월1일',
    #   '6월30일'
    # ])
    #
    # self.to_and_from_test(string, year, month, day, pattern, language=language, cur_year=cur_year, cur_month=cur_month, cur_day=cur_day, cur_day_of_week=cur_day_of_week, zeros=False, no_year=True)

  def to_and_from_test(self, string, year, month, day, pattern, **kwargs):
    string_out = date_to_string(year, month, day, pattern, **kwargs)
    self.assertAllEqual(string_out, string)

    year_out, month_out, day_out = string_to_date(string, pattern, **kwargs)
    self.assertAllEqual(year_out, year)
    self.assertAllEqual(month_out, month)
    self.assertAllEqual(day_out, day)

  def test_delta_day(self):
    pattern = np.array(['delta_day'] * 8)
    language = np.array(['en'] * 4 + ['ja'] * 4)
    cur_year = np.array([2021, 1960, 2030, 1745] * 2)
    cur_month = np.array([3, 12, 1, 6] * 2)
    cur_day = np.array([27, 31, 1, 30] * 2)
    cur_day_of_week = np.array([5, 5, 1, 2] * 2)

    year = np.array([2021, 1961, 2030, 1745] * 2)
    month = np.array([3, 1, 1, 7] * 2)
    day = np.array([26, 1, 3, 1] * 2)

    num = np.array([0, 0, 0, 1] * 2)

    string = np.array([
      'yesterday',
      'tomorrow',
      'day after tomorrow',
      'Tomorrow',
      '昨日',
      '明日',
      '明後日',
      'あす'
    ])

    self.to_and_from_test(string, year, month, day, pattern, language=language, cur_year=cur_year, cur_month=cur_month, cur_day=cur_day, cur_day_of_week=cur_day_of_week, num=num)
    #
    # language = 'ja'
    # string = np.array([
    #   '昨日',
    #   '明日',
    #   '明後日',
    #   'あす'
    # ])
    #
    # self.to_and_from_test(string, year, month, day, pattern, language=language, cur_year=cur_year, cur_month=cur_month, cur_day=cur_day, cur_day_of_week=cur_day_of_week, num=num)

  def test_delta_day_of_week(self):
    pattern = np.array(['delta_day_of_week']*12)
    language = np.array(['en'] * 8 + ['ja'] * 4)
    cur_year = np.array([2021, 1960, 2030, 1745]*3)
    cur_month = np.array([3, 12, 1, 6]*3)
    cur_day = np.array([27, 31, 1, 30]*3)
    cur_day_of_week = np.array([5, 5, 1, 2] * 3)

    year = np.array([2021, 1961, 2029, 1745] * 3)
    month = np.array([3, 1, 12, 7] * 3)
    day = np.array([29, 9, 28, 2] * 3)

    next = np.array(['', 'next', 'last', 'this'] * 3)
    abbr = np.array([False] * 4 + [True] * 4 + [False] * 4)
    cap = np.array([False] * 4 + [True] * 4 + [False] * 4)
    string = np.array([
      'monday',
      'next monday',
      'last friday',
      'this friday',
      'Mon',
      'Next Mon',
      'Last Fri',
      'This Fri',
      '月曜日',
      '次の月曜日',
      '先週の金曜日',
      '今度の金曜日'
    ])
    self.to_and_from_test(string, year, month, day, pattern, language=language, cur_year=cur_year, cur_month=cur_month, cur_day=cur_day, cur_day_of_week=cur_day_of_week, next=next, cap=cap, abbr=abbr)
    #
    # abbr = np.array([True, True, True, True])
    # cap = np.array([True, True, True, True])
    # string = np.array([
    #   'Mon',
    #   'Next Mon',
    #   'Last Fri',
    #   'This Fri'
    # ])
    #
    # self.to_and_from_test(string, year, month, day, pattern, language=language, cur_year=cur_year, cur_month=cur_month, cur_day=cur_day, cur_day_of_week=cur_day_of_week, cap=cap, abbr=abbr, next=next)
    #
    # language = 'ja'
    # string = np.array([
    #   '月曜日',
    #   '次の月曜日',
    #   '先週の金曜日',
    #   '今度の金曜日'
    # ])

    # self.to_and_from_test(string, year, month, day, pattern, language=language, cur_year=cur_year, cur_month=cur_month, cur_day=cur_day, cur_day_of_week=cur_day_of_week, next=next)

if __name__ == '__main__':
  tf.test.main()
