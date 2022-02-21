from typing import Optional
import numpy as np
import pandas as pd
import datetime
import mlp.data_generation.maps as m


def time_to_string(
  hour: np.ndarray,
  minute: np.ndarray,
  second: np.ndarray,
  pattern: np.ndarray,
  language: Optional[np.ndarray] = None,
  **kwargs
  ) -> np.ndarray:
  """
  Convert a ints hours minutes and seconds to some string pattern.

  Parameters
  ----------
  hour: the hour values
  minute: the minute values
  second: the second values
  pattern: which string pattern to use
  language: the language of the string

  Returns
  -------
  string: the string representation of the time

  """
  if language is None:
    language = np.array([''] * len(hour))

  indices = np.argsort(pattern)
  inverse_indices = np.argsort(indices)
  sorted_pattern = pattern[indices]

  uniques, group_indices = np.unique(sorted_pattern, return_index=True)

  all_arrays = {
    'hour': hour, 'minute': minute, 'second': second,
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
    string_group = pattern_functions[unique]['time_to_string'](
      **group_kwargs
    )
    string_groups.append(string_group)

  strings = np.concatenate(string_groups)
  if len(group_indices[1:]):
    strings = strings[inverse_indices]
  return strings


def string_to_time(
  string: np.ndarray,
  pattern: np.ndarray,
  language: Optional[np.ndarray] = None,
  **kwargs
  ) -> (np.ndarray, np.ndarray, np.ndarray):
  """
  Convert a string of some time pattern into arrays of hours, minutes and seconds.

  Parameters
  ----------
  string: the string representation of the time
  language: the language of the string
  pattern: which string pattern to use

  Returns
  -------
  hour: the hour values
  minute: the minute values
  second: the second values
  """
  if language is None:
    language = np.array([''] * len(string))

  indices = np.argsort(pattern)
  inverse_indices = np.argsort(indices)
  sorted_pattern = pattern[indices]

  uniques, group_indices = np.unique(sorted_pattern, return_index=True)

  all_arrays = {
    'string': string,
    'language': language
  }
  all_arrays.update(kwargs)

  array_groups = {}
  for array_key, array in all_arrays.items():
    if len(group_indices[1:]):
      array_groups[array_key] = np.split(array[indices], group_indices[1:])
    else:
      array_groups[array_key] = [array]
  hour_groups = []
  minute_groups = []
  second_groups = []
  for group_num, unique in enumerate(uniques):
    group_kwargs = {k: v[group_num] for k, v in array_groups.items()}

    hour_group, minute_group, second_group = pattern_functions[unique]['string_to_time'](
      **group_kwargs
    )
    hour_groups.append(hour_group)
    minute_groups.append(minute_group)
    second_groups.append(second_group)

  hour = np.concatenate(hour_groups)
  minute = np.concatenate(minute_groups)
  second = np.concatenate(second_groups)

  if len(group_indices[1:]):
    hour = hour[inverse_indices]
    minute = minute[inverse_indices]
    second = second[inverse_indices]

  return hour, minute, second


def _extract_digits(string, lang, digit_type, use_none=False):
  number = ''
  found = True
  while found:
    found = False
    for word in m.string_to_integer[lang]:
      if digit_type != m.string_to_integer[lang][word]['word_type']:
        continue

      if word == string[:len(word)]:
        string = string[len(word):]
        num = m.string_to_integer[lang][word]['integer']
        number = number + str(num)
        found = True
        break
  if number == '':
    if use_none:
      return None, string
    return 0, string
  return int(number), string


def digit_hour_string_to_time(
  string: np.ndarray,
  language: Optional[np.ndarray] = None,
  digit_type: Optional[np.ndarray] = None,
  am_pm_type: Optional[np.ndarray] = None,
  hour_word: Optional[np.ndarray] = None,
  zeros: Optional[np.ndarray] = None,
  **kwargs
  ) -> (np.ndarray, np.ndarray, np.ndarray):
  """
  Convert a string of some time pattern into arrays of hours, minutes and seconds.

  Parameters
  ----------
  string: the string representation of the time
  language: the language of the string
  digit_type: How to represent the digits, e.g. full-width, half-width, word, etc.
  am_pm_type: the string denoting some am, pm, in the evening, 朝, etc value.
  hour_word: the string separating the hour from the rest of the time. e.g. ':', '時', ' ', etc.
  zeros: Whether or not to have leading zeros for single digit hours and minutes.

  Returns
  -------
  hour: the hour values
  minute: the minute values
  second: the second values
  """
  if digit_type is None:
    digit_type = np.array(['digit']*len(string))
  if am_pm_type is None:
    am_pm_type = np.array(['']*len(string))
  if hour_word is None:
    hour_word = np.array([':']*len(string))
  if zeros is None:
    zeros = np.array([False]*len(string))

  second = np.array([0]*len(string))
  minute = np.array([0]*len(string))

  hour = []
  for s, lang,  ampm, dt, hw, z in zip(string, language, am_pm_type, digit_type, hour_word, zeros):
    if ampm not in s:
      hour.append(None)
      minute.append(None)
      continue
    delta = m.am_pm_to_tuple[lang][ampm][0]
    s = s.strip()
    if lang == 'en':
      h, s = _extract_digits(s, lang, dt)
    elif lang in ('ja', 'zh_hant', 'zh_hant', 'ko'):
      s = s[len(ampm):]
      h, s = _extract_digits(s, lang, dt)
    else:
      raise ValueError("{} not a supported language".format(lang))
    h = h + delta
    hour.append(h)

  hour = np.array(hour)
  return (hour, minute, second)


def digit_hour_time_to_string(
  hour: np.ndarray,
  minute: np.ndarray,
  second: np.ndarray,
  language: Optional[np.ndarray] = None,
  digit_type: Optional[np.ndarray] = None,
  am_pm_type: Optional[np.ndarray] = None,
  hour_word: Optional[np.ndarray] = None,
  zeros: Optional[np.ndarray] = None,
  **kwargs
  ) -> np.ndarray:
  """
  Convert a ints hours minutes and seconds to some string pattern.

  Parameters
  ----------
  hour: the hour values
  minute: the minute values
  second: the second values
  language: the language of the string
  digit_type: How to represent the digits, e.g. full-width, half-width, word, etc.
  am_pm_type: the string denoting some am, pm, in the evening, 朝, etc value.
  hour_word: the string separating the hour from the rest of the time. e.g. ':', '時', ' ', etc.
  short_cut_word: the string denoting some specific number of minutes. e.g. '半', 'half past', etc.
  zeros: Whether or not to have leading zeros for single digit hours and minutes.

  Returns
  -------
  string: the string representation of the time

  """
  string = []
  for h, _, __, lang,  ampm, dt, hw, z in zip(hour, minute, second, language, am_pm_type, digit_type, hour_word, zeros):

    delta = m.am_pm_to_tuple[lang][ampm][0]
    hour_str = ''
    if h > 12:
      h = h - delta
    if h < 10 and z:
      hour_str = m.integer_to_string[lang][0][dt]

    for sub_str in str(h):
      hour_str = hour_str + m.integer_to_string[lang][int(sub_str)][dt]

    if lang == 'en':
      hour_str = (hour_str + hw).strip()
      hour_str = (hour_str + ampm).strip()
    elif lang in ('ja', 'zh_hans', 'zh_hant', 'ko'):
      hour_str = (ampm + hour_str).strip()
      hour_str = (hour_str + hw).strip()
    else:
      raise ValueError("{} not a supported language".format(lang))

    string.append(hour_str)
  return np.array(string)


def digit_hour_minute_string_to_time(
  string: np.ndarray,
  language: Optional[np.ndarray] = None,
  digit_type: Optional[np.ndarray] = None,
  am_pm_type: Optional[np.ndarray] = None,
  hour_word: Optional[np.ndarray] = None,
  minute_word: Optional[np.ndarray] = None,
  short_cut_word: Optional[np.ndarray] = None,
  zeros: Optional[np.ndarray] = None,
  **kwargs
  ) -> (np.ndarray, np.ndarray, np.ndarray):
  """
  Convert a string of some time pattern into arrays of hours, minutes and seconds.

  Parameters
  ----------
  string: the string representation of the time
  language: the language of the string
  digit_type: How to represent the digits, e.g. full-width, half-width, word, etc.
  am_pm_type: the string denoting some am, pm, in the evening, 朝, etc value.
  hour_word: the string separating the hour from the rest of the time. e.g. ':', '時', ' ', etc.
  minute_word: the string separating the minute from the rest of the time. e.g. ':', '分', '', etc.
  short_cut_word: the string denoting some specific number of minutes. e.g. '半', 'half past', etc.
  zeros: Whether or not to have leading zeros for single digit hours and minutes.

  Returns
  -------
  hour: the hour values
  minute: the minute values
  second: the second values
  """
  if digit_type is None:
    digit_type = np.array(['digit']*len(string))
  if am_pm_type is None:
    am_pm_type = np.array(['']*len(string))
  if hour_word is None:
    hour_word = np.array(['']*len(string))
  if minute_word is None:
    minute_word = np.array(['']*len(string))
  if short_cut_word is None:
    short_cut_word = np.array(['']*len(string))
  if zeros is None:
    zeros = np.array([False]*len(string))

  second = np.array([0]*len(string))
  minute = np.array([0]*len(string))

  hour = []
  minute = []
  for s, lang,  ampm, dt, mw, scw, z in zip(string, language, am_pm_type, digit_type,  minute_word, short_cut_word, zeros):

    ampm_delta = m.am_pm_to_tuple[lang][ampm]
    scw_delta = m.short_cut_words[lang][scw]

    if ampm not in s or scw not in s or mw not in s:
      hour.append(None)
      minute.append(None)
      continue
    s = s.strip()
    if lang == 'en':
      s = s[len(scw):]
      h, s = _extract_digits(s, lang, dt, use_none=True)
    elif lang in ('ja', 'zh_hant', 'zh_hant', 'ko'):
      s = s[len(ampm):]
      h, s = _extract_digits(s, lang, dt, use_none=True)
    else:
      raise ValueError("{} not a supported language".format(lang))

    h = h + ampm_delta[0] + scw_delta[0]
    hour.append(h)

    for hw in m.hour_words[lang]:
      s = s.replace(hw, '')

    min, s = _extract_digits(s, lang, dt)
    min = min + ampm_delta[1] + scw_delta[1]
    minute.append(min)
  hour = np.array(hour)
  return (hour, minute, second)


def digit_hour_minute_time_to_string(
  hour: np.ndarray,
  minute: np.ndarray,
  second: np.ndarray,
  language: Optional[np.ndarray] = None,
  digit_type: Optional[np.ndarray] = None,
  am_pm_type: Optional[np.ndarray] = None,
  hour_word: Optional[np.ndarray] = None,
  minute_word: Optional[np.ndarray] = None,
  short_cut_word: Optional[np.ndarray] = None,
  zeros: Optional[np.ndarray] = None,
  **kwargs
  ) -> np.ndarray:
  """
  Convert a ints hours minutes and seconds to some string pattern.

  Parameters
  ----------
  hour: the hour values
  minute: the minute values
  second: the second values
  language: the language of the string
  digit_type: How to represent the digits, e.g. full-width, half-width, word, etc.
  am_pm_type: the string denoting some am, pm, in the evening, 朝, etc value.
  hour_word: the string separating the hour from the rest of the time. e.g. ':', '時', ' ', etc.
  minute_word: the string separating the minute from the rest of the time. e.g. ':', '分', '', etc.
  short_cut_word: the string denoting some specific number of minutes. e.g. '半', 'half past', etc.
  zeros: Whether or not to have leading zeros for single digit hours and minutes.

  Returns
  -------
  string: the string representation of the time

  """
  string = []
  for h, min, _, lang,  ampm, dt, hw, mw, scw, z in zip(hour, minute, second, language, am_pm_type, digit_type, hour_word, minute_word, short_cut_word, zeros):

    delta = m.am_pm_to_tuple[lang][ampm][0]
    if h > 12:
      h = h - delta
    full_str = ''

    hour_str = ''
    if h < 10 and z and dt not in ('word', 'alt_word_0'):
      hour_str = m.integer_to_string[lang][0][dt]

    for sub_str in str(h):
      if lang in ('ja', 'ko', 'zh_hant', 'zh_hans') or dt != 'word':
        hour_str = hour_str + m.integer_to_string[lang][int(sub_str)][dt]
      else:
        hour_str = hour_str + ' ' + m.integer_to_string[lang][int(sub_str)][dt]

    minute_str = ''
    if min < 10 and z:
      minute_str = m.integer_to_string[lang][0][dt]

    for sub_str in str(min):
      minute_str = minute_str + m.integer_to_string[lang][int(sub_str)][dt]

    if lang == 'en':
      minute_str = (minute_str + mw).strip()
      if scw != '':
        if hw == ':':
          hw = ''
        hour_str = (hour_str + hw).strip()
        full_str = scw + hour_str + ampm
      else:
        hour_str = (hour_str + hw).strip()
        full_str = full_str + hour_str + minute_str + ampm

    elif lang in ('ja', 'zh_hans', 'zh_hant', 'ko'):
      hour_str = (ampm + hour_str).strip()
      minute_str = (minute_str + mw).strip()
      full_str = hour_str + minute_str + scw

      if scw != '':
        if hw == ':':
          hw = ''
        hour_str = (hour_str + hw).strip()
        full_str = hour_str + scw
      else:
        hour_str = (hour_str + hw).strip()
        full_str = hour_str + minute_str

    else:
      raise ValueError("{} not a supported language".format(lang))

    string.append(full_str)
  return np.array(string)


pattern_functions = {
  'digit_hour_minute': {
    'time_to_string': digit_hour_minute_time_to_string,
    'string_to_time': digit_hour_minute_string_to_time},
  'digit_hour': {
    'time_to_string': digit_hour_time_to_string,
    'string_to_time': digit_hour_string_to_time},

}
