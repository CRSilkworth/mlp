from typing import Optional
import numpy as np
import mlp.data_generation.maps as m


def integer_to_string(
  integer: np.ndarray,
  pattern: np.ndarray,
  language: Optional[np.ndarray] = None,
  **kwargs
  ) -> np.ndarray:
  """
  Convert integers to some string pattern.

  Parameters
  ----------
  integer: the integers
  pattern: which string pattern to use
  language: the language of the string

  Returns
  -------
  string: the string representation of the integer

  """
  if language is None:
    language = np.array([''] * len(integer))
  indices = np.argsort(pattern)
  inverse_indices = np.argsort(indices)
  sorted_pattern = pattern[indices]

  uniques, group_indices = np.unique(sorted_pattern, return_index=True)

  all_arrays = {
    'integer': integer,
    'language': language
  }
  all_arrays.update(kwargs)

  array_groups = {}
  for array_key, array in all_arrays.items():
    if len(group_indices[1:]):
      array_groups[array_key] = np.split(array[indices], group_indices[1:])
    else:
      array_groups[array_key] = [array]

  string_groups = []
  for group_num, unique in enumerate(uniques):
    group_kwargs = {k: v[group_num] for k, v in array_groups.items()}
    string_group = pattern_functions[unique]['integer_to_string'](
      **group_kwargs
    )
    string_groups.append(string_group)

  strings = np.concatenate(string_groups)

  if len(group_indices[1:]):
    strings = strings[inverse_indices]

  return strings


def string_to_integer(
  string: np.ndarray,
  pattern: np.ndarray,
  language: Optional[np.ndarray] = None,
  **kwargs
  ) -> np.ndarray:
  """
  Convert strings to some integers.

  Parameters
  ----------
  string: the string representation of the integer
  pattern: which string pattern to use
  language: the language of the string

  Returns
  -------
  integer: the integers

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

  integer_groups = []
  for group_num, unique in enumerate(uniques):
    group_kwargs = {k: v[group_num] for k, v in array_groups.items()}
    integer_group = pattern_functions[unique]['string_to_integer'](
      **group_kwargs
    )
    integer_groups.append(integer_group)

  integer = np.concatenate(integer_groups)

  if len(group_indices[1:]):
    integer = integer[inverse_indices]

  return integer


def cast_integer_to_string(
  integer: np.ndarray,
  **kwargs
  ) -> np.ndarray:
  """
  Convert integers to some string pattern.

  Parameters
  ----------
  integer: the integers

  Returns
  -------
  string: the string representation of the integer

  """
  return integer.astype(str)


def cast_string_to_integer(
  string: np.ndarray,
  **kwargs
  ) -> np.ndarray:
  """
  Convert strings to some integers.

  Parameters
  ----------
  string: the string representation of the integer

  Returns
  -------
  integer: the integers

  """
  return string.astype(int)


def comma_integer_to_string(
  integer: np.ndarray,
  **kwargs
  ) -> np.ndarray:
  """
  Convert integers to some string pattern.

  Parameters
  ----------
  integer: the integers

  Returns
  -------
  string: the string representation of the integer

  """
  formatter = np.vectorize("{:,}".format)
  return formatter(integer)

def comma_string_to_integer(
  string: np.ndarray,
  **kwargs
  ) -> np.ndarray:
  """
  Convert strings to some integers.

  Parameters
  ----------
  string: the string representation of the integer

  Returns
  -------
  integer: the integers

  """

  string = np.char.replace(string, ',', '')
  return string.astype(int)

def rec_txt_to_int(words, lang):
  if not words:
    return 0
  if len(words) == 1:
    return m.word_to_digit[lang][words[0]]['integer']
  max_num = None
  max_val = None
  for num, word in enumerate(words):
    val = m.word_to_digit[lang][word]['integer']
    if max_val is None or max_val < val:
      max_val = val
      max_num = num
  left = rec_txt_to_int(words[:max_num], lang) or 1
  right = rec_txt_to_int(words[max_num + 1:], lang)
  return left * max_val + right


def txt_to_int(textnum, lang, word_type='word'):
  if lang == 'en':
    textnum = textnum.replace('-', ' ')
    words = textnum.split()
  elif lang in ('ja', 'ko', 'zh_hans', 'zh_hant'):
    words = list(textnum)
  return rec_txt_to_int(words, lang)


def word_string_to_integer(
  string: np.ndarray,
  language: np.ndarray,
  **kwargs
  ) -> np.ndarray:
  """
  Convert strings to some integers.

  Parameters
  ----------
  string: the string representation of the integer
  language: the language of the string

  Returns
  -------
  integer: the integers

  """
  integer = []
  for s, lang in zip(string, language):
    integer.append(txt_to_int(s, lang))
  integer = np.array(integer)
  return integer

def int_to_txt(i, lang, word_type='word'):
    """
    Convert an integer in to it's word representation.

    int_to_txt(i: integer) -> string
    """
    if i < 0:
        return _join('negative', _int_to_txt_pos(-i, lang, word_type))
    if i == 0:
        return m.digit_to_word[lang][0][word_type]
    return _int_to_txt_pos(i, lang, word_type)


def _divide(dividend, divisor, magnitude, lang, word_type='word'):

    divided = _int_to_txt_pos(dividend // divisor, lang, word_type)
    remainder = _int_to_txt_pos(dividend % divisor, lang, word_type)
    return _join(
        divided,
        magnitude,
        remainder,
        lang=lang
    )


def _join(*args, lang=None):
    if lang == 'en':
      return ' '.join(filter(bool, args))
    elif lang in ('ja', 'ko', 'zh_hans', 'zh_hant'):
      return ''.join(filter(bool, args))
    else:
      raise ValueError("unsupported language: {}".format(lang))


def _int_to_txt_pos(i, lang, word_type='word'):
  if i == 0:
    return ''
  if lang == 'en':
    if i < 20:
      return m.digit_to_word[lang][i][word_type]
    if i < 100:
      remainder = m.digit_to_word[lang][i % 10][word_type]
      joined = _join(m.digit_to_word[lang][(i // 10)*10][word_type], remainder if i % 10 != 0 else '', lang=lang)
      return joined
    if i < 1000:
      return _divide(i, 100, 'hundred', lang, word_type)

    for digit, d in m.digit_to_word[lang].items():
      word = d[word_type]
      if digit < 1000:
        continue

      if i < digit:
        break
      last_number = digit
      last_word = word
    # return _divide(i, 1000**number, word, lang)
    divided = _divide(i, last_number, last_word, lang, word_type)
    return divided
  elif lang in ('ja', 'ko', 'zh_hans', 'zh_hant'):
    if i < 10:
      return m.digit_to_word[lang][i][word_type]

    last_number = 10
    last_name = m.digit_to_word[lang][10][word_type]
    for digit, d in m.digit_to_word[lang].items():
      word = d[word_type]

      if digit < 13:
        continue

      if i < digit:
        break
      last_number = digit
      last_name = word
    return _divide(i, last_number, last_name, lang, word_type)


def word_integer_to_string(
  integer: np.ndarray,
  language: np.ndarray,
  **kwargs
  ) -> np.ndarray:
  """
  Convert integers to some string pattern.

  Parameters
  ----------
  integer: the integers
  language: the language of the string

  Returns
  -------
  string: the string representation of the integer

  """
  string = []
  for i, lang in zip(integer, language):
    s = int_to_txt(i, lang)
    string.append(s)
  string = np.array(string)
  return string


def cap_word_string_to_integer(
  string: np.ndarray,
  language: np.ndarray,
  word_type: Optional[np.ndarray] = None,
  **kwargs
  ) -> np.ndarray:
  """
  Convert strings to some integers.

  Parameters
  ----------
  string: the string representation of the integer
  language: the language of the string

  Returns
  -------
  integer: the integers

  """
  if word_type is None:
    word_type = np.array(['word']*len(string))

  string = np.char.lower(string)

  integer = []
  for s, lang, wt in zip(string, language, word_type):
    s = s.replace(' and ', ' ')
    integer.append(txt_to_int(s, lang, word_type=wt))
  integer = np.array(integer)
  return integer


def cap_word_integer_to_string(
  integer: np.ndarray,
  language: np.ndarray,
  **kwargs
  ) -> np.ndarray:
  """
  Convert integers to some string pattern.

  Parameters
  ----------
  integer: the integers
  language: the language of the string

  Returns
  -------
  string: the string representation of the integer

  """
  string = []
  for i, lang in zip(integer, language):
    s = int_to_txt(i, lang)
    s = s.capitalize()
    string.append(s)
  string = np.array(string)
  return string
  # return np.char.capitalize(string)


def digit_word_string_to_integer(
  string: np.ndarray,
  language: np.ndarray,
  digit_type: Optional[np.ndarray] = None,
  **kwargs
  ) -> np.ndarray:
  """
  Convert strings to some integers.

  Parameters
  ----------
  string: the string representation of the integer
  language: the language of the string
  digit_type: digit_type: How to represent the digits, e.g. full-width, half-width, word, etc.

  Returns
  -------
  integer: the integers

  """
  if digit_type is None:
    digit_type = np.array(['digit'] * len(string))

  string = np.char.lower(string)

  integer = []
  for s, lang, dt in zip(string, language, digit_type):
    for num in m.digit_to_word[lang]:
      sub = m.digit_to_word[lang][num][dt]
      s = s.replace(sub, str(num))
    s = s.replace(' ', '')
    integer.append(s)
  integer = np.array(integer).astype(int)
  return integer


def digit_word_integer_to_string(
  integer: np.ndarray,
  language: np.ndarray,
  digit_type: Optional[np.ndarray] = None,
  **kwargs
  ) -> np.ndarray:
  """
  Convert integers to some string pattern.

  Parameters
  ----------
  integer: the integers
  language: the language of the string
  digit_type: How to represent the digits, e.g. full-width, half-width, word, etc.

  Returns
  -------
  string: the string representation of the integer

  """
  if digit_type is None:
    digit_type = np.array(['digit'] * len(integer))
  integer = integer.astype(str)
  string = []
  for row_num, (s, lang, dt) in enumerate(zip(integer, language, digit_type)):
    for num in m.digit_to_word[lang]:
      sub = m.digit_to_word[lang][num][dt]
      if lang == 'en' and dt == 'word':
        sub_str = ' ' + str(sub)
      else:
        sub_str = str(sub)
      s = s.replace(str(num), sub_str)
      s = s.strip()
    string.append(s)
  string = np.array(string)
  return string


def flt_wrd_to_int(string, language):
  if language == 'en':

    words = string.split()
    if len(words) != 2:
      raise ValueError("not of the form <number><scale_word> got: {}".format(words))
    val = float(words[0]) * m.word_to_digit[language][words[-1]]['integer']
  elif language in ('ja', 'ko', 'zh_hans', 'zh_hant'):
    words = list(string)
    val = float(''.join(words[:-1])) * m.word_to_digit[language][words[-1]]['integer']
  else:
    raise ValueError("unsupported language: {}".format(language))

  if not val.is_integer():
    raise ValueError("{} is not integer ".format(string))

  return int(val)


def float_word_string_to_integer(
  string: np.ndarray,
  language: np.ndarray,
  **kwargs
  ) -> np.ndarray:
  """
  Convert strings to some integers.

  Parameters
  ----------
  string: the string representation of the integer
  language: the language of the string

  Returns
  -------
  integer: the integers

  """
  integer = []
  for s, lang in zip(string, language):
    integer.append(flt_wrd_to_int(s, lang))
  integer = np.array(integer)
  return integer


def int_to_flt_wrd(integer, language, word_type='word'):
  max_word = None
  max_val = None
  for val, d in m.digit_to_word[language].items():
    word = d[word_type]

    if val > integer:
      continue

    if max_val is None or val > max_val:
      max_val = val
      max_word = word

  if max_val > 10:
    num_part = integer / float(max_val)
  else:
    num_part = integer
    max_word = ''

  if int(num_part) == num_part:
    num_part = int(num_part)

  if language == 'en':
    return str(num_part) + ' ' + max_word

  elif language in ('ja', 'ko', 'zh_hans', 'zh_hant'):
    return str(num_part) + max_word
  else:
    raise ValueError("unsupported language: {}".format(language))


def float_word_integer_to_string(
  integer: np.ndarray,
  language: np.ndarray,
  **kwargs
  ) -> np.ndarray:
  """
  Convert integers to some string pattern.

  Parameters
  ----------
  integer: the integers
  language: the language of the string

  Returns
  -------
  string: the string representation of the integer

  """
  string = []
  for i, lang in zip(integer, language):
    s = int_to_flt_wrd(i, lang)
    string.append(s)
  string = np.array(string)
  return string


pattern_functions = {
  'cast': {
    'integer_to_string': cast_integer_to_string,
    'string_to_integer': cast_string_to_integer},

  'comma': {
    'integer_to_string': comma_integer_to_string,
    'string_to_integer': comma_string_to_integer},

  'word': {
    'integer_to_string': word_integer_to_string,
    'string_to_integer': word_string_to_integer},

  'cap_word': {
    'integer_to_string': cap_word_integer_to_string,
    'string_to_integer': cap_word_string_to_integer},

  # 'digit_word': {
  #   'integer_to_string': digit_word_integer_to_string,
  #   'string_to_integer': digit_word_string_to_integer},

  'float_word': {
    'integer_to_string': float_word_integer_to_string,
    'string_to_integer': float_word_string_to_integer},

}
