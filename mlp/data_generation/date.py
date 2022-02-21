from typing import Optional
import numpy as np
import pandas as pd
import datetime

string_to_month = {}
string_to_month['en'] = {
  'january': 1,
  'february': 2,
  'march': 3,
  'april': 4,
  'may': 5,
  'june': 6,
  'july': 7,
  'august': 8,
  'september': 9,
  'october': 10,
  'november': 11,
  'december': 12
}

month_to_string = {}
month_to_string['ja'] = {1: 'ー月', 2: '二月', 3: '三月', 4: '四月', 5: '五月', 6: '六月', 7: '七月', 8: '八月', 9: '九月', 10: '十月', 11: '十一月', 12: '十二月'}
month_to_string['zh_hans'] = month_to_string['ja']
month_to_string['zh_hant'] = month_to_string['ja']
month_to_string['ko'] = {1: '일', 2: '이', 3: '삼', 4: '사', 5: '오', 6: '육', 7: '칠', 8: '팔', 9: '구', 10: '십', 11: '십일', 12: '십이'}
for lang in month_to_string:
  string_to_month[lang] = {v: k for k, v in month_to_string[lang].items()}

for lang in string_to_month:
  month_to_string[lang] = {v: k for k, v in string_to_month[lang].items()}


ja_era_to_offset = {'文亀': 1500, '永正': 1503, '大永': 1520, '享禄': 1527, '天文': 1531, '弘治': 1554, '永禄': 1557, '元亀': 1569, '天正': 1572, '文禄': 1591, '慶長': 1595, '元和': 1614, '寛永': 1623, '正保': 1643, '慶安': 1647, '承応': 1651, '明暦': 1654, '万治': 1657, '寛文': 1660, '延宝': 1672, '天和': 1680, '貞享': 1683, '元禄': 1687, '宝永': 1703, '正徳': 1710, '享保': 1715, '元文': 1735, '寛保': 1740, '延享': 1743, '寛延': 1747, '宝暦': 1750, '明和': 1763, '安永': 1771, '天明': 1780, '寛政': 1788, '享和': 1800, '文化': 1803, '文政': 1817, '天保': 1829, '弘化': 1843, '嘉永': 1847, '安政': 1853, '万延': 1859, '文久': 1860, '元治': 1863, '慶応': 1864, '明治': 1867, '大正': 1911, '昭和': 1925, '平成': 1988, '令和': 2018}
offset_to_ja_era = {v: k for k, v in ja_era_to_offset.items()}


def era_to_year(num, era_name):
  return ja_era_to_offset[era_name] + num


def year_to_era(year):
  for era, offset in list(ja_era_to_offset.items())[::-1]:
    if year > offset + 1:
      break
  return (year - offset, era)


def date_to_tuple(date):
  return (
    date.astype('datetime64[Y]').astype(int) + 1970,
    date.astype('datetime64[M]').astype(int) % 12 + 1,
    (date - date.astype('datetime64[M]') + 1).astype(int)
  )


def date_to_string(
  year: np.ndarray,
  month: np.ndarray,
  day: np.ndarray,
  pattern: np.ndarray,
  cur_year: Optional[np.ndarray] = None,
  cur_month: Optional[np.ndarray] = None,
  cur_day: Optional[np.ndarray] = None,
  cur_day_of_week: Optional[np.ndarray] = None,
  language: Optional[np.ndarray] = None,
  **kwargs
  ) -> np.ndarray:
  """
  Convert a ints years months and days to some string pattern.

  Parameters
  ----------
  year: the year values
  month: the month values
  day: the day values
  cur_year: the current year values
  cur_month: the current month values
  cur_day: the current day values
  cur_day_of_week: the current day of the week values (1-7)
  pattern: which string pattern to use
  language: the language of the string

  Returns
  -------
  string: the string representation of the date

  """
  if language is None:
    language = np.array([''] * len(year))
  if cur_year is None:
    cur_year = np.array([''] * len(year))
  if cur_month is None:
    cur_month = np.array([''] * len(year))
  if cur_day is None:
    cur_day = np.array([''] * len(year))

  indices = np.argsort(pattern)
  inverse_indices = np.argsort(indices)
  sorted_pattern = pattern[indices]

  uniques, group_indices = np.unique(sorted_pattern, return_index=True)

  all_arrays = {
    'year': year, 'month': month, 'day': day,
    'cur_year': cur_year, 'cur_month': cur_month, 'cur_day': cur_day,
    'language': language
  }
  all_arrays.update(kwargs)

  array_groups = {}
  for array_key, array in all_arrays.items():
    array_groups[array_key] = np.split(array[indices], group_indices[1:])
    if len(group_indices[1:]):
      array_groups[array_key] = np.split(array[indices], group_indices[1:])
    else:
      array_groups[array_key] = [array]

  string_groups = []
  for group_num, unique in enumerate(uniques):
    group_kwargs = {k: v[group_num] for k, v in array_groups.items()}
    string_group = pattern_functions[unique]['date_to_string'](
      **group_kwargs
    )
    string_groups.append(string_group)

  strings = np.concatenate(string_groups)
  if len(group_indices[1:]):
    strings = strings[inverse_indices]
  return strings


def string_to_date(
  string: np.ndarray,
  pattern: np.ndarray,
  cur_year: Optional[np.ndarray] = None,
  cur_month: Optional[np.ndarray] = None,
  cur_day: Optional[np.ndarray] = None,
  cur_day_of_week: Optional[np.ndarray] = None,
  language: Optional[np.ndarray] = None,
  **kwargs
  ) -> (np.ndarray, np.ndarray, np.ndarray):
  """
  Convert a string of some date pattern into arrays of years, months and days.

  Parameters
  ----------
  string: the string representation of the date
  language: the language of the string
  cur_year: the current year values
  cur_month: the current month values
  cur_day: the current day values
  cur_day_of_week: the current day of the week values (1-7)

  Returns
  -------
  year: the year values
  month: the month values
  day: the day values
  """
  if language is None:
    language = np.array([''] * len(string))
  if cur_year is None:
    cur_year = np.array([''] * len(string))
  if cur_month is None:
    cur_month = np.array([''] * len(string))
  if cur_day is None:
    cur_day = np.array([''] * len(string))

  indices = np.argsort(pattern)
  inverse_indices = np.argsort(indices)
  sorted_pattern = pattern[indices]

  uniques, group_indices = np.unique(sorted_pattern, return_index=True)

  all_arrays = {
    'string': string,
    'cur_year': cur_year, 'cur_month': cur_month, 'cur_day': cur_day, 'cur_day_of_week': cur_day_of_week,
    'language': language
  }
  all_arrays.update(kwargs)

  array_groups = {}
  for array_key, array in all_arrays.items():
    if len(group_indices[1:]):
      array_groups[array_key] = np.split(array[indices], group_indices[1:])
    else:
      array_groups[array_key] = [array]
  year_groups = []
  month_groups = []
  day_groups = []
  for group_num, unique in enumerate(uniques):
    group_kwargs = {k: v[group_num] for k, v in array_groups.items()}

    year_group, month_group, day_group = pattern_functions[unique]['string_to_date'](
      **group_kwargs
    )
    year_groups.append(year_group)
    month_groups.append(month_group)
    day_groups.append(day_group)

  year = np.concatenate(year_groups)
  month = np.concatenate(month_groups)
  day = np.concatenate(day_groups)

  if len(group_indices[1:]):
    year = year[inverse_indices]
    month = month[inverse_indices]
    day = day[inverse_indices]

  return year, month, day

def zip_many(**kwargs):

  keys = list(kwargs.keys())
  values = list(kwargs.values())

  for row in zip(*values):
    d = {k: r for k, r in zip(keys, row)}
    yield d


def ymd_date_to_string(
  year: np.ndarray,
  month: np.ndarray,
  day: np.ndarray,
  cur_year: Optional[np.ndarray] = None,
  cur_month: Optional[np.ndarray] = None,
  cur_day: Optional[np.ndarray] = None,
  cur_day_of_week: Optional[np.ndarray] = None,
  language: Optional[np.ndarray] = None,
  zeros: Optional[np.ndarray] = None,
  delimiter: Optional[np.ndarray] = None,
  no_year: Optional[np.ndarray] = None,
  **kwargs
  ) -> np.ndarray:
  """
  Convert a ints years months and days to some string pattern.

  Parameters
  ----------
  year: the year values
  month: the month values
  day: the day values
  cur_year: the current year values
  cur_month: the current month values
  cur_day: the current day values
  cur_day_of_week: the current day of the week values (1-7)
  language: the language of the string
  zeros: Whether or not to include leading zeros on single digit days and months
  delimiter: the delimiter separating year, month and day
  no_year: whether or not to exclude the year (assumed to be current year)


  Returns
  -------
  string: the string representation of the date

  """
  if zeros is None:
    zeros = np.array([True] * len(year))
  if delimiter is None:
    delimiter = np.array(['-'] * len(year))
  if no_year is None:
    no_year = np.array([True] * len(year))

  iterator = zip_many(
    year=year,
    month=month,
    day=day,
    cur_year=cur_year,
    cur_month=cur_month,
    cur_day=cur_day,
    language=language,
    zeros=zeros,
    delimiter=delimiter,
    no_year=no_year
  )

  string = []
  for r in iterator:
    r['month'] = str(r['month'])
    r['day'] = str(r['day'])

    if r['zeros']:
      r['month'] = '0' + r['month'] if len(r['month']) < 2 else r['month']
      r['day'] = '0' + r['day'] if len(r['day']) < 2 else r['day']

    if not r['no_year']:
      s = str(r['year']) + r['delimiter']
    else:
      s = ''

    s = s + r['month'] + r['delimiter'] + r['day']
    string.append(s)

  string = np.array(string)
  return string


def many_format(string, delimiter, formats):
  uniques, indices = np.unique(delimiter, return_index=True)
  string_groups = np.split(string, indices[1:])
  concat_strings = []
  for string_group, unique in zip(string_groups, uniques):
    s = pd.to_datetime(string_group, format=formats[0] + unique + formats[1] + unique + formats[2]).astype(str)
    concat_strings.append(s.to_numpy())
  concat_strings = np.concatenate(concat_strings)
  return concat_strings


def ymd_string_to_date(
  string: np.ndarray,
  cur_year: Optional[np.ndarray] = None,
  cur_month: Optional[np.ndarray] = None,
  cur_day: Optional[np.ndarray] = None,
  cur_day_of_week: Optional[np.ndarray] = None,
  language: Optional[np.ndarray] = None,
  zeros: Optional[np.ndarray] = None,
  delimiter: Optional[np.ndarray] = None,
  no_year: Optional[np.ndarray] = np.ndarray,
  **kwargs
  ) -> (np.ndarray, np.ndarray, np.ndarray):
  """
  Convert a string of some date pattern into arrays of years, months and days.

  Parameters
  ----------
  string: the string representation of the date
  cur_year: the current year values
  cur_month: the current month values
  cur_day: the current day values
  cur_day_of_week: the current day of the week values (1-7)
  language: the language of the string
  zeros: Whether or not to include leading zeros on single digit days and months
  delimiter: the delimiter separating year, month and day
  no_year: whether or not to exclude the year (assumed to be current year)

  Returns
  -------
  year: the year values
  month: the month values
  day: the day values

  """
  if zeros is None:
    zeros = np.array([True] * len(string))
  if delimiter is None:
    delimiter = np.array(['-'] * len(string))
  if no_year is None:
    no_year = np.array([True] * len(string))
  year_str = np.char.add(delimiter[no_year], cur_year[no_year].astype(str))
  string[no_year] = np.char.add(string[no_year], year_str)

  string = many_format(string, delimiter, ('%Y', '%m', '%d'))
  string = string.astype('datetime64')

  return date_to_tuple(string)


def mdy_date_to_string(
  year: np.ndarray,
  month: np.ndarray,
  day: np.ndarray,
  cur_year: Optional[np.ndarray] = None,
  cur_month: Optional[np.ndarray] = None,
  cur_day: Optional[np.ndarray] = None,
  cur_day_of_week: Optional[np.ndarray] = None,
  language: Optional[np.ndarray] = None,
  zeros: Optional[bool] = True,
  delimiter: Optional[str] = '-',
  no_year: Optional[bool] = False,
  **kwargs
  ) -> np.ndarray:
  """
  Convert a ints years months and days to some string pattern.

  Parameters
  ----------
  year: the year values
  month: the month values
  day: the day values
  cur_year: the current year values
  cur_month: the current month values
  cur_day: the current day values
  cur_day_of_week: the current day of the week values (1-7)
  language: the language of the string
  zeros: Whether or not to include leading zeros on single digit days and months
  delimiter: the delimiter separating year, month and day
  no_year: whether or not to exclude the year (assumed to be current year)
  Returns
  -------
  string: the string representation of the date

  """
  if zeros is None:
    zeros = np.array([True] * len(year))
  if delimiter is None:
    delimiter = np.array(['-'] * len(year))
  if no_year is None:
    no_year = np.array([True] * len(year))

  iterator = zip_many(
    year=year,
    month=month,
    day=day,
    cur_year=cur_year,
    cur_month=cur_month,
    cur_day=cur_day,
    language=language,
    zeros=zeros,
    delimiter=delimiter,
    no_year=no_year
  )

  string = []
  for r in iterator:
    r['month'] = str(r['month'])
    r['day'] = str(r['day'])

    if r['zeros']:
      r['month'] = '0' + r['month'] if len(r['month']) < 2 else r['month']
      r['day'] = '0' + r['day'] if len(r['day']) < 2 else r['day']

    s = r['month'] + r['delimiter'] + r['day']

    if not r['no_year']:
      s = s + r['delimiter'] + str(r['year'])

    string.append(s)

  string = np.array(string)

  return string


def mdy_string_to_date(
  string: np.ndarray,
  cur_year: Optional[np.ndarray] = None,
  cur_month: Optional[np.ndarray] = None,
  cur_day: Optional[np.ndarray] = None,
  cur_day_of_week: Optional[np.ndarray] = None,
  language: Optional[np.ndarray] = None,
  zeros: Optional[np.ndarray] = None,
  delimiter: Optional[np.ndarray] = None,
  no_year: Optional[np.ndarray] = None,
  **kwargs
  ) -> (np.ndarray, np.ndarray, np.ndarray):
  """
  Convert a string of some date pattern into arrays of years, months and days.

  Parameters
  ----------
  string: the string representation of the date
  cur_year: the current year values
  cur_month: the current month values
  cur_day: the current day values
  cur_day_of_week: the current day of the week values (1-7)
  language: the language of the string
  zeros: Whether or not to include leading zeros on single digit days and months
  delimiter: the delimiter separating year, month and day
  no_year: whether or not to exclude the year (assumed to be current year)

  Returns
  -------
  year: the year values
  month: the month values
  day: the day values

  """
  if zeros is None:
    zeros = np.array([True] * len(string))
  if delimiter is None:
    delimiter = np.array(['-'] * len(string))
  if no_year is None:
    no_year = np.array([True] * len(string))

  year_str = np.char.add(delimiter[no_year], cur_year[no_year].astype(str))
  string[no_year] = np.char.add(string[no_year], year_str)

  date = many_format(string, delimiter, ('%m', '%d', '%Y'))
  date = date.astype('datetime64')

  return date_to_tuple(date)


def dmy_date_to_string(
  year: np.ndarray,
  month: np.ndarray,
  day: np.ndarray,
  cur_year: Optional[np.ndarray] = None,
  cur_month: Optional[np.ndarray] = None,
  cur_day: Optional[np.ndarray] = None,
  cur_day_of_week: Optional[np.ndarray] = None,
  language: Optional[np.ndarray] = None,
  zeros: Optional[np.ndarray] = None,
  delimiter: Optional[np.ndarray] = None,
  no_year: Optional[np.ndarray] = None,
  **kwargs
  ) -> np.ndarray:
  """
  Convert a ints years months and days to some string pattern.

  Parameters
  ----------
  year: the year values
  month: the month values
  day: the day values
  cur_year: the current year values
  cur_month: the current month values
  cur_day: the current day values
  cur_day_of_week: the current day of the week values (1-7)
  language: the language of the string
  zeros: Whether or not to include leading zeros on single digit days and months
  no_year: whether or not to exclude the year (assumed to be current year)

  Returns
  -------
  string: the string representation of the date

  """
  if zeros is None:
    zeros = np.array([True] * len(year))
  if delimiter is None:
    delimiter = np.array(['-'] * len(year))
  if no_year is None:
    no_year = np.array([True] * len(year))

  iterator = zip_many(
    year=year,
    month=month,
    day=day,
    cur_year=cur_year,
    cur_month=cur_month,
    cur_day=cur_day,
    language=language,
    zeros=zeros,
    delimiter=delimiter,
    no_year=no_year
  )

  string = []
  for r in iterator:
    r['month'] = str(r['month'])
    r['day'] = str(r['day'])
    if r['zeros']:
      r['month'] = '0' + r['month'] if len(r['month']) < 2 else r['month']
      r['day'] = '0' + r['day'] if len(r['day']) < 2 else r['day']

    s = r['day'] + r['delimiter'] + r['month']

    if not r['no_year']:
      s = s + r['delimiter'] + str(r['year'])

    string.append(s)

  string = np.array(string)

  return string


def dmy_string_to_date(
  string: np.ndarray,
  cur_year: Optional[np.ndarray] = None,
  cur_month: Optional[np.ndarray] = None,
  cur_day: Optional[np.ndarray] = None,
  cur_day_of_week: Optional[np.ndarray] = None,
  language: Optional[np.ndarray] = None,
  zeros: Optional[np.ndarray] = None,
  delimiter: Optional[np.ndarray] = None,
  no_year: Optional[np.ndarray] = None,
  **kwargs
  ) -> (np.ndarray, np.ndarray, np.ndarray):
  """
  Convert a string of some date pattern into arrays of years, months and days.

  Parameters
  ----------
  string: the string representation of the date
  cur_year: the current year values
  cur_month: the current month values
  cur_day: the current day values
  cur_day_of_week: the current day of the week values (1-7)
  language: the language of the string
  zeros: Whether or not to include leading zeros on single digit days and months
  postfix: whether or not to include the postfix after the day
  no_year: whether or not to exclude the year (assumed to be current year)
  delimiter: the delimiter separating year, month and day

  Returns
  -------
  year: the year values
  month: the month values
  day: the day values

  """
  if zeros is None:
    zeros = np.array([True] * len(string))
  if delimiter is None:
    delimiter = np.array(['-'] * len(string))
  if no_year is None:
    no_year = np.array([True] * len(string))

  year_str = np.char.add(delimiter[no_year], cur_year[no_year].astype(str))
  string[no_year] = np.char.add(string[no_year], year_str)

  string = many_format(string, delimiter, ('%d', '%m', '%Y'))
  string = string.astype('datetime64')

  return date_to_tuple(string)


num_to_postfix = {
  '0': 'th',
  '1': 'st',
  '2': 'nd',
  '3': 'rd'
}
num_to_postfix.update({str(i): 'th' for i in range(4, 10)})


def add_postfix(day):
  str_day = str(day)
  num = str_day[-1]

  return str_day + num_to_postfix[num]


def word_dy_date_to_string(
  year: np.ndarray,
  month: np.ndarray,
  day: np.ndarray,
  cur_year: Optional[np.ndarray] = None,
  cur_month: Optional[np.ndarray] = None,
  cur_day: Optional[np.ndarray] = None,
  cur_day_of_week: Optional[np.ndarray] = None,
  language: Optional[np.ndarray] = None,
  zeros: Optional[bool] = False,
  postfix: Optional[bool] = False,
  no_year: Optional[bool] = False,
  **kwargs
  ) -> np.ndarray:
  """
  Convert a ints years months and days to some string pattern.

  Parameters
  ----------
  year: the year values
  month: the month values
  day: the day values
  cur_year: the current year values
  cur_month: the current month values
  cur_day: the current day values
  cur_day_of_week: the current day of the week values (1-7)
  language: the language of the string
  zeros: Whether or not to include leading zeros on single digit days and months
  postfix: whether or not to include the postfix after the day
  no_year: whether or not to exclude the year (assumed to be current year)

  Returns
  -------
  string: the string representation of the date

  """
  if zeros is None:
    zeros = np.array([True] * len(year))
  if no_year is None:
    no_year = np.array([True] * len(year))

  iterator = zip_many(
    year=year,
    month=month,
    day=day,
    cur_year=cur_year,
    cur_month=cur_month,
    cur_day=cur_day,
    language=language,
    zeros=zeros,
    no_year=no_year,
    postfix=postfix
  )

  string = []
  for r in iterator:
    r['month'] = str(r['month'])
    r['day'] = str(r['day'])
    if r['zeros']:
      r['month'] = '0' + r['month'] if len(r['month']) < 2 else r['month']
      r['day'] = '0' + r['day'] if len(r['day']) < 2 else r['day']

    r['month'] = month_to_string[r['language']][int(r['month'])]
    r['month'] = r['month'].capitalize()

    if r['postfix']:
      r['day'] = add_postfix(r['day'])

    s = r['month'] + ' ' + r['day']

    if not r['no_year']:
      s = s + ', ' + str(r['year'])

    string.append(s)

  string = np.array(string)

  return string


def word_dy_string_to_date(
  string: np.ndarray,
  cur_year: Optional[np.ndarray] = None,
  cur_month: Optional[np.ndarray] = None,
  cur_day: Optional[np.ndarray] = None,
  cur_day_of_week: Optional[np.ndarray] = None,
  language: Optional[np.ndarray] = None,
  zeros: Optional[bool] = False,
  postfix: Optional[bool] = False,
  no_year: Optional[bool] = False,
  **kwargs
  ) -> (np.ndarray, np.ndarray, np.ndarray):
  """
  Convert a string of some date pattern into arrays of years, months and days.

  Parameters
  ----------
  string: the string representation of the date
  cur_year: the current year values
  cur_month: the current month values
  cur_day: the current day values
  cur_day_of_week: the current day of the week values (1-7)
  language: the language of the string
  zeros: Whether or not to include leading zeros on single digit days and months
  postfix: whether or not to include the postfix after the day
  no_year: whether or not to exclude the year (assumed to be current year)

  Returns
  -------
  year: the year values
  month: the month values
  day: the day values

  """
  subarray = string[((language == 'en') & postfix)]
  subarray = np.char.replace(subarray, 'st', '')
  subarray = np.char.replace(subarray, 'nd', '')
  subarray = np.char.replace(subarray, 'rd', '')
  subarray = np.char.replace(subarray, 'th', '')

  string[((language == 'en') & postfix)] = subarray
  year_str = np.char.add(', ', cur_year[no_year].astype(str))
  string[no_year] = np.char.add(string[no_year], year_str)

  string = np.vectorize(lambda x: datetime.datetime.strptime(x, '%B %d, %Y'))(string)
  string = np.vectorize(np.datetime64)(string).astype('datetime64[D]')

  return date_to_tuple(string)


def d_word_y_date_to_string(
  year: np.ndarray,
  month: np.ndarray,
  day: np.ndarray,
  cur_year: Optional[np.ndarray] = None,
  cur_month: Optional[np.ndarray] = None,
  cur_day: Optional[np.ndarray] = None,
  cur_day_of_week: Optional[np.ndarray] = None,
  language: Optional[np.ndarray] = None,
  zeros: Optional[np.ndarray] = None,
  postfix: Optional[np.ndarray] = None,
  no_year: Optional[np.ndarray] = None,
  with_of: Optional[np.ndarray] = None,
  **kwargs
  ) -> np.ndarray:
  """
  Convert a ints years months and days to some string pattern.

  Parameters
  ----------
  year: the year values
  month: the month values
  day: the day values
  cur_year: the current year values
  cur_month: the current month values
  cur_day: the current day values
  cur_day_of_week: the current day of the week values (1-7)
  language: the language of the string
  zeros: Whether or not to include leading zeros on single digit days and months
  postfix: whether or not to include the postfix after the day
  no_year: whether or not to exclude the year (assumed to be current year)
  with_of: wether or not include 'of {month}'

  Returns
  -------
  string: the string representation of the date

  """
  if zeros is None:
    zeros = np.array([True] * len(year))
  if no_year is None:
    no_year = np.array([True] * len(year))
  if with_of is None:
    with_of = np.array([False] * len(year))

  iterator = zip_many(
    year=year,
    month=month,
    day=day,
    cur_year=cur_year,
    cur_month=cur_month,
    cur_day=cur_day,
    language=language,
    zeros=zeros,
    no_year=no_year,
    postfix=postfix,
    with_of=with_of,
  )

  string = []
  for r in iterator:
    r['month'] = str(r['month'])
    r['day'] = str(r['day'])
    if r['zeros']:
      r['month'] = '0' + r['month'] if len(r['month']) < 2 else r['month']
      r['day'] = '0' + r['day'] if len(r['day']) < 2 else r['day']

    r['month'] = month_to_string[r['language']][int(r['month'])]
    r['month'] = str(r['month'])
    r['month'] = r['month'].capitalize()

    if r['postfix']:
      r['day'] = add_postfix(r['day'])

    if r['with_of'] and r['language'] == 'en':
      s = r['day'] + ' of ' + r['month']

    else:
      s = r['day'] + ' ' + r['month']

    if not r['no_year']:
      s = s + ', ' + str(r['year'])

    string.append(s)

  string = np.array(string)

  return string


def d_word_y_string_to_date(
  string: np.ndarray,
  cur_year: Optional[np.ndarray] = None,
  cur_month: Optional[np.ndarray] = None,
  cur_day: Optional[np.ndarray] = None,
  cur_day_of_week: Optional[np.ndarray] = None,
  language: Optional[np.ndarray] = None,
  zeros: Optional[np.ndarray] = None,
  postfix: Optional[np.ndarray] = None,
  no_year: Optional[np.ndarray] = None,
  **kwargs
  ) -> (np.ndarray, np.ndarray, np.ndarray):
  """
  Convert a string of some date pattern into arrays of years, months and days.

  Parameters
  ----------
  string: the string representation of the date
  cur_year: the current year values
  cur_month: the current month values
  cur_day: the current day values
  cur_day_of_week: the current day of the week values (1-7)
  language: the language of the string
  zeros: Whether or not to include leading zeros on single digit days and months
  postfix: whether or not to include the postfix after the day
  no_year: whether or not to exclude the year (assumed to be current year)

  Returns
  -------
  year: the year values
  month: the month values
  day: the day values

  """
  if zeros is None:
    zeros = np.array([False] * len(string))
  if postfix is None:
    postfix = np.array([False] * len(string))
  if no_year is None:
    no_year = np.array([False] * len(string))

  subarray = string[((language == 'en') & postfix)]
  subarray = np.char.replace(subarray, 'st', '')
  subarray = np.char.replace(subarray, 'nd', '')
  subarray = np.char.replace(subarray, 'rd', '')
  subarray = np.char.replace(subarray, 'th', '')

  string[((language == 'en') & postfix)] = subarray
  year_str = np.char.add(', ', cur_year[no_year].astype(str))
  string[no_year] = np.char.add(string[no_year], year_str)

  string = np.vectorize(lambda x: datetime.datetime.strptime(x, '%d %B, %Y'))(string)

  string = np.vectorize(np.datetime64)(string).astype('datetime64[D]')

  return date_to_tuple(string)


def ja_date_to_string(
  year: np.ndarray,
  month: np.ndarray,
  day: np.ndarray,
  cur_year: Optional[np.ndarray] = None,
  cur_month: Optional[np.ndarray] = None,
  cur_day: Optional[np.ndarray] = None,
  cur_day_of_week: Optional[np.ndarray] = None,
  language: Optional[np.ndarray] = None,
  zeros: Optional[np.ndarray] = None,
  delimiter: Optional[np.ndarray] = None,
  no_year: Optional[np.ndarray] = None,
  era: Optional[np.ndarray] = None,
  **kwargs
  ) -> np.ndarray:
  """
  Convert a ints years months and days to some string pattern.

  Parameters
  ----------
  year: the year values
  month: the month values
  day: the day values
  cur_year: the current year values
  cur_month: the current month values
  cur_day: the current day values
  cur_day_of_week: the current day of the week values (1-7)
  language: the language of the string
  no_year: whether or not to exclude the year (assumed to be current year)
  era: whether or not to write the year in the japanese era format

  Returns
  -------
  string: the string representation of the date

  """
  if zeros is None:
    zeros = np.array([True] * len(year))
  if delimiter is None:
    delimiter = np.array(['-'] * len(year))
  if no_year is None:
    no_year = np.array([True] * len(year))
  if era is None:
    era = np.array([False] * len(year))

  iterator = zip_many(
    year=year,
    month=month,
    day=day,
    cur_year=cur_year,
    cur_month=cur_month,
    cur_day=cur_day,
    language=language,
    zeros=zeros,
    delimiter=delimiter,
    no_year=no_year,
    era=era
  )

  string = []
  for r in iterator:
    if r['era']:
      num, era_name = year_to_era(r['year'])
      r['year'] = era_name + str(num)

    r['month'] = str(r['month'])
    r['day'] = str(r['day'])
    if r['zeros']:
      r['month'] = '0' + r['month'] if len(r['month']) < 2 else r['month']
      r['day'] = '0' + r['day'] if len(r['day']) < 2 else r['day']

    if not r['no_year']:
      s = str(r['year']) + '年'
    else:
      s = ''

    s = s + r['month'] + '月' + r['day'] + '日'
    string.append(s)

  string = np.array(string)
  return string


def _ja_string_to_date(string, era=False):

  if '年' in string:
    first_split = string.split('年')
    year = first_split[0]
    month = first_split[1].split('月')[0]
    day = first_split[1].split('月')[1].replace('日', '')
  else:
    month = string.split('月')[0]
    day = string.split('月')[1].replace('日', '')
    year = None

  if era:
    era_name = year[:2]
    num = year[2:]
    year = str(era_to_year(num, era_name))

  if len(month) < 2:
    month = '0' + month
  if len(day) < 2:
    day = '0' + day

  if year is not None:
    return '-'.join([year, month, day])
  else:
    return '-'.join([month, day])


def ja_string_to_date(
  string: np.ndarray,
  cur_year: Optional[np.ndarray] = None,
  cur_month: Optional[np.ndarray] = None,
  cur_day: Optional[np.ndarray] = None,
  cur_day_of_week: Optional[np.ndarray] = None,
  language: Optional[np.ndarray] = None,
  no_year: Optional[np.ndarray] = None,
  era: Optional[np.ndarray] = None,
  **kwargs
  ) -> (np.ndarray, np.ndarray, np.ndarray):
  """
  Convert a string of some date pattern into arrays of years, months and days.

  Parameters
  ----------
  string: the string representation of the date
  cur_year: the current year values
  cur_month: the current month values
  cur_day: the current day values
  cur_day_of_week: the current day of the week values (1-7)
  language: the language of the string
  no_year: whether or not to exclude the year (assumed to be current year)
  era: whether or not to write the year in the japanese era format

  Returns
  -------
  year: the year values
  month: the month values
  day: the day values

  """
  if era is None:
    era = np.array([False] * len(string))
  if no_year is None:
    no_year = np.array([False] * len(string))

  string = np.vectorize(_ja_string_to_date)(string)
  year_str = np.char.add(cur_year[no_year].astype(str), '-')
  full_strings = np.char.add(year_str, string[no_year]).tolist()
  new_string = []
  for num, (s, ny) in enumerate(zip(string, no_year)):
    if ny:
      new_string.append(full_strings.pop(0))
    else:
      new_string.append(s)

  # string[no_year] = np.char.add(year_str, string[no_year])
  # string[no_year] = new_strings
  new_string = np.array(new_string).astype('datetime64')

  return date_to_tuple(new_string)


def ko_date_to_string(
  year: np.ndarray,
  month: np.ndarray,
  day: np.ndarray,
  cur_year: Optional[np.ndarray] = None,
  cur_month: Optional[np.ndarray] = None,
  cur_day: Optional[np.ndarray] = None,
  cur_day_of_week: Optional[np.ndarray] = None,
  language: Optional[np.ndarray] = None,
  zeros: Optional[np.ndarray] = None,
  no_year: Optional[np.ndarray] = None,
  delimiter: Optional[np.ndarray] = None,
  **kwargs
  ) -> np.ndarray:
  """
  Convert a ints years months and days to some string pattern.

  Parameters
  ----------
  year: the year values
  month: the month values
  day: the day values
  cur_year: the current year values
  cur_month: the current month values
  cur_day: the current day values
  cur_day_of_week: the current day of the week values (1-7)
  language: the language of the string
  zeros: Whether or not to include leading zeros on single digit days and months
  no_year: whether or not to exclude the year (assumed to be current year)
  delimiter: the delimiter separating year, month and day

  Returns
  -------
  string: the string representation of the date

  """
  if zeros is None:
    zeros = np.array([True] * len(year))
  if delimiter is None:
    delimiter = np.array(['-'] * len(year))
  if no_year is None:
    no_year = np.array([True] * len(year))

  iterator = zip_many(
    year=year,
    month=month,
    day=day,
    cur_year=cur_year,
    cur_month=cur_month,
    cur_day=cur_day,
    language=language,
    zeros=zeros,
    delimiter=delimiter,
    no_year=no_year
  )

  string = []
  for r in iterator:
    r['month'] = str(r['month'])
    r['day'] = str(r['day'])
    if r['zeros']:
      r['month'] = '0' + r['month'] if len(r['month']) < 2 else r['month']
      r['day'] = '0' + r['day'] if len(r['day']) < 2 else r['day']

    if not r['no_year']:
      s = str(r['year']) + '년'
    else:
      s = ''

    s = s + r['month'] + '월' + r['day'] + '일'
    string.append(s)

  string = np.array(string)
  return string


def _ko_string_to_date(string):
  if '년' in string:
    first_split = string.split('년')
    year = first_split[0]
    month = first_split[1].split('월')[0]
    day = first_split[1].split('월')[1].replace('일', '')
  else:
    month = string.split('월')[0]
    day = string.split('월')[1].replace('일', '')
    year = None

  if len(month) < 2:
    month = '0' + month
  if len(day) < 2:
    day = '0' + day

  if year is not None:
    return '-'.join([year, month, day])
  else:
    return '-'.join([month, day])


def ko_string_to_date(
  string: np.ndarray,
  cur_year: Optional[np.ndarray] = None,
  cur_month: Optional[np.ndarray] = None,
  cur_day: Optional[np.ndarray] = None,
  cur_day_of_week: Optional[np.ndarray] = None,
  language: Optional[np.ndarray] = None,
  no_year: Optional[bool] = False,
  **kwargs
  ) -> (np.ndarray, np.ndarray, np.ndarray):
  """
  Convert a string of some date pattern into arrays of years, months and days.

  Parameters
  ----------
  string: the string representation of the date
  cur_year: the current year values
  cur_month: the current month values
  cur_day: the current day values
  cur_day_of_week: the current day of the week values (1-7)
  language: the language of the string
  no_year: whether or not to exclude the year (assumed to be current year)

  Returns
  -------
  year: the year values
  month: the month values
  day: the day values

  """
  string = np.vectorize(_ko_string_to_date)(string)
  year_str = np.char.add(cur_year[no_year].astype(str), '-')
  string[no_year] = np.char.add(year_str, string[no_year])

  string = string.astype('datetime64')

  return date_to_tuple(string)

delta_day = {
  'en': {'tomorrow': 1, 'Tomorrow': 1, 'yesterday': -1, 'Yesterday': -1, 'day before yesterday': -2, 'day after tomorrow': 2, 'today': 0},
  'ja': {'明日': 1, 'あす': 1, 'あした': 1, '昨日': -1, 'きのう': -1, '一昨日': -2, 'おととい': -2, '明後日': 2, 'あさって': 2, '今日': 0},
  'ko': {'내일': 1, '어제': -1, '모레': 2, '그저께': -2, '오늘': 0},
  'zh_hant': {'明天': 1, '明日': 1, '前天': -2, '昨天': -1, '後天': 2, '今天': 0, },
  'zh_hans': {'明天': 1, '明日': 1, '前天': -2, '昨天': -1, '後天': 2, '今天': 0}
}

delta_day_inv = {}
for lang in ['en', 'ja', 'zh_hans', 'zh_hant', 'ko']:
  delta_day_inv[lang] = {}
  for word, delta in delta_day[lang].items():
    delta_day_inv[lang].setdefault(delta, [])
    delta_day_inv[lang][delta].append(word)


def delta_day_date_to_string(
  year: np.ndarray,
  month: np.ndarray,
  day: np.ndarray,
  cur_year: Optional[np.ndarray] = None,
  cur_month: Optional[np.ndarray] = None,
  cur_day: Optional[np.ndarray] = None,
  cur_day_of_week: Optional[np.ndarray] = None,
  language: Optional[np.ndarray] = None,
  num: Optional[np.ndarray] = None,
  **kwargs
  ) -> np.ndarray:
  """
  Convert a ints years months and days to some string pattern.

  Parameters
  ----------
  year: the year values
  month: the month values
  day: the day values
  cur_year: the current year values
  cur_month: the current month values
  cur_day: the current day values
  cur_day_of_week: the current day of the week values (1-7)
  language: the language of the string
  num: if there is more than one 'next' word for the given input, which to select

  Returns
  -------
  string: the string representation of the date

  """
  if num is None:
    num = np.array([0]*len(year))

  cur_date = tuple_to_datetime(cur_year, cur_month, cur_day)
  date = tuple_to_datetime(year, month, day)

  num_days = (date - cur_date).astype(int)

  iterator = zip_many(
    year=year,
    month=month,
    day=day,
    cur_year=cur_year,
    cur_month=cur_month,
    cur_day=cur_day,
    language=language,
    num=num,
    num_days=num_days
  )

  string = []
  for r in iterator:
    string.append(delta_day_inv[r['language']][r['num_days']][r['num']])

  string = np.array(string)
  return string


def tuple_to_datetime(cur_year, cur_month, cur_day):
  cur_month = cur_month.astype(str)
  cur_day = cur_day.astype(str)
  f = np.vectorize(lambda x: '0' + x if len(x) < 2 else x)
  cur_month = f(cur_month)
  cur_day = f(cur_day)

  string = np.char.add(cur_year.astype(str), '-')
  string = np.char.add(string, cur_month.astype(str))
  string = np.char.add(string, '-')
  string = np.char.add(string, cur_day.astype(str))

  return string.astype('datetime64[D]')


def delta_day_string_to_date(
  string: np.ndarray,
  cur_year: Optional[np.ndarray] = None,
  cur_month: Optional[np.ndarray] = None,
  cur_day: Optional[np.ndarray] = None,
  cur_day_of_week: Optional[np.ndarray] = None,
  language: Optional[np.ndarray] = None,
  **kwargs
  ) -> (np.ndarray, np.ndarray, np.ndarray):
  """
  Convert a string of some date pattern into arrays of years, months and days.

  Parameters
  ----------
  string: the string representation of the date
  cur_year: the current year values
  cur_month: the current month values
  cur_day: the current day values
  cur_day_of_week: the current day of the week values (1-7)
  language: the language of the string

  Returns
  -------
  year: the year values
  month: the month values
  day: the day values

  """
  date = tuple_to_datetime(cur_year, cur_month, cur_day)

  deltas = []
  for word, lang in zip(string, language):
    deltas.append(delta_day[lang][word])
  deltas = np.array(deltas).astype('timedelta64')
  date = date + deltas

  return date_to_tuple(date)


day_of_week_index = {
  'en': {
    0: ['monday', 'mon'],
    1: ['tuesday', 'tue'],
    2: ['wednesday', 'wed'],
    3: ['thursday', 'thu'],
    4: ['friday', 'fri'],
    5: ['saturday', 'sat'],
    6: ['sunday', 'sun'],

  },
  'ja': {
    0: ['月曜日', '月曜'],
    1: ['火曜日', '火曜'],
    2: ['水曜日', '水曜'],
    3: ['木曜日', '木曜'],
    4: ['金曜日', '金曜'],
    5: ['土曜日', '土曜'],
    6: ['日曜日', '日曜'],


  },
  'ko': {
    0: ['월요일', '월'],
    1: ['화요일', '화'],
    2: ['수요일', '수'],
    3: ['목요일', '목'],
    4: ['금요일', '금'],
    5: ['토요일', '토'],
    6: ['일요일', '일'],

  },

  'zh_hans': {
    0: ['星期一', '周一'],
    1: ['星期二', '周二'],
    2: ['星期三', '周三'],
    3: ['星期四', '周四'],
    4: ['星期五', '周五'],
    5: ['星期六', '周六'],
    6: ['星期日', '周日'],

  },

  'zh_hant': {
    0: ['星期一', '週一'],
    1: ['星期二', '週二'],
    2: ['星期三', '週三'],
    3: ['星期四', '週四'],
    4: ['星期五', '週五'],
    5: ['星期六', '週六'],
    6: ['星期日', '週日'],

  }
}
next_words = {
  'this': {
    'en': ["this "],
    'ja': ["今度の"],
    'ko': ["이번주"],
    'zh_hans': ["这个"],
    'zh_hant': ["这个"],
  },
  'next': {
    'en': ["next "],
    'ja': ["次の", "来週の"],
    'ko': ["다음"],
    'zh_hans': ["下周"],
    'zh_hant': ["下週"],
  },
  'last': {
    'en': ["last "],
    'ja': ["先週の"],
    'ko': ["지난"],
    'zh_hans': ["上个"],
    'zh_hant': ["上個"],
  }
}


def delta_day_of_week_date_to_string(
  year: np.ndarray,
  month: np.ndarray,
  day: np.ndarray,
  cur_year: Optional[np.ndarray] = None,
  cur_month: Optional[np.ndarray] = None,
  cur_day: Optional[np.ndarray] = None,
  cur_day_of_week: Optional[np.ndarray] = None,
  language: Optional[np.ndarray] = None,
  cap: Optional[np.ndarray] = None,
  next: Optional[np.ndarray] = None,
  abbr: Optional[np.ndarray] = None,
  num: Optional[np.ndarray] = None,
  **kwargs
  ) -> np.ndarray:
  """
  Convert a ints years months and days to some string pattern.

  Parameters
  ----------
  year: the year values
  month: the month values
  day: the day values
  cur_year: the current year values
  cur_month: the current month values
  cur_day: the current day values
  cur_day_of_week: the current day of the week values (1-7)
  language: the language of the string
  cap: whether or not to capitalize the first word
  next: whether the word is 'next', 'last', 'the day after', etc.
  abbr: whether to abbreviate or not
  num: if there is more than one 'next' word for the given input, which to select

  Returns
  -------
  string: the string representation of the date

  """
  if cap is None:
    cap = np.array([False]*len(year))
  if next is None:
    next = np.array(['']*len(year))
  if abbr is None:
    abbr = np.array([False]*len(year))
  if num is None:
    num = np.array([0]*len(year))

  cur_date = tuple_to_datetime(cur_year, cur_month, cur_day)
  date = tuple_to_datetime(year, month, day)
  dayofweek = pd.Series(date).dt.dayofweek
  num_days = (date - cur_date).astype(int)

  iterator = zip_many(
    year=year,
    month=month,
    day=day,
    cur_year=cur_year,
    cur_month=cur_month,
    cur_day=cur_day,
    language=language,
    cap=cap,
    next=next,
    abbr=abbr,
    num=num,
    num_days=num_days,
    dayofweek=dayofweek
  )

  string = []
  for r in iterator:
    if r['next'] in next_words:
      word = next_words[r['next']][r['language']][r['num']]
    else:
      word = ''
    r['dayofweek'] = day_of_week_index[r['language']][r['dayofweek']][r['abbr']]

    if r['cap']:
      word = word.capitalize()
      r['dayofweek'] = r['dayofweek'].capitalize()

    string.append(word + r['dayofweek'])

  string = np.array(string)

  return string


def delta_day_of_week_string_to_date(
  string: np.ndarray,
  cur_year: Optional[np.ndarray] = None,
  cur_month: Optional[np.ndarray] = None,
  cur_day: Optional[np.ndarray] = None,
  cur_day_of_week: Optional[np.ndarray] = None,
  language: Optional[np.ndarray] = None,
  cap: Optional[np.ndarray] = None,
  next: Optional[np.ndarray] = None,
  abbr: Optional[np.ndarray] = None,
  num: Optional[np.ndarray] = None,
  **kwargs
  ) -> (np.ndarray, np.ndarray, np.ndarray):
  """
  Convert a string of some date pattern into arrays of years, months and days.

  Parameters
  ----------
  string: the string representation of the date
  cur_year: the current year values
  cur_month: the current month values
  cur_day: the current day values
  cur_day_of_week: the current day of the week values (1-7)
  language: the language of the string
  cap: whether or not to capitalize the first word
  next: whether the word is 'next', 'last', 'the day after', etc.
  abbr: whether to abbreviate or not
  num: if there is more than one 'next' word for the given input, which to select

  Returns
  -------
  year: the year values
  month: the month values
  day: the day values

  """
  if cap is None:
    cap = np.array([False]*len(string))
  if next is None:
    next = np.array(['']*len(string))
  if abbr is None:
    abbr = np.array([False]*len(string))
  if num is None:
    num = np.array([0]*len(string))

  cur_date = tuple_to_datetime(cur_year, cur_month, cur_day)
  # print('*'*100)
  # print(cur_day_of_week)
  iterator = zip_many(
    string=string,
    cur_year=cur_year,
    cur_month=cur_month,
    cur_day=cur_day,
    cur_day_of_week=cur_day_of_week,
    language=language,
    cap=cap,
    next=next,
    abbr=abbr,
    num=num,
  )

  delta = []
  for r in iterator:
    if r['next'] in next_words:
      word = next_words[r['next']][r['language']][r['num']]
    else:
      word = ''
    r['string'] = r['string'].lower().replace(word, '').strip()
    for indx, tup in day_of_week_index[r['language']].items():
      if r['string'] == tup[int(r['abbr'])]:
        dow = indx
        break
    day_diff = (dow - r['cur_day_of_week']) % 7
    if r['next'] == 'this':
      day_diff = day_diff if day_diff > 0 else 7
    elif r['next'] == 'next':
      day_diff = day_diff + 7
    elif r['next'] == 'last':
      day_diff = -1 * ((r['cur_day_of_week'] - dow) % 7)
    delta.append(day_diff)

  delta = np.array(delta)
  date = cur_date + delta

  return date_to_tuple(date)


pattern_functions = {
  'ymd': {
    'date_to_string': ymd_date_to_string,
    'string_to_date': ymd_string_to_date},
  'mdy': {
    'date_to_string': mdy_date_to_string,
    'string_to_date': mdy_string_to_date},

  'dmy': {
    'date_to_string': dmy_date_to_string,
    'string_to_date': dmy_string_to_date},

  'word_dy': {
    'date_to_string': word_dy_date_to_string,
    'string_to_date': word_dy_string_to_date},
  'd_word_y': {
    'date_to_string': d_word_y_date_to_string,
    'string_to_date': d_word_y_string_to_date},

  'ja': {
    'date_to_string': ja_date_to_string,
    'string_to_date': ja_string_to_date},
  'ko': {
    'date_to_string': ko_date_to_string,
    'string_to_date': ko_string_to_date},

  'delta_day': {
    'date_to_string': delta_day_date_to_string,
    'string_to_date': delta_day_string_to_date},

  'delta_day_of_week': {
    'date_to_string': delta_day_of_week_date_to_string,
    'string_to_date': delta_day_of_week_string_to_date},
}
