import mlp.data_generation.integer as integer_dg
from typing import Optional
import numpy as np
import mlp.data_generation.maps as m
import fractions

def number_to_string(
  number: np.ndarray,
  pattern: np.ndarray,
  language: Optional[np.ndarray] = None,
  **kwargs
  ) -> np.ndarray:
  if language is None:
    language = np.array([''] * len(number))
  indices = np.argsort(pattern)
  inverse_indices = np.argsort(indices)
  sorted_pattern = pattern[indices]

  uniques, group_indices = np.unique(sorted_pattern, return_index=True)

  all_arrays = {
    'number': number,
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
    string_group = pattern_functions[unique]['number_to_string'](
      **group_kwargs
    )
    string_groups.append(string_group)

  strings = np.concatenate(string_groups)

  if len(group_indices[1:]):
    strings = strings[inverse_indices]

  return strings


def string_to_number(
  string: np.ndarray,
  pattern: np.ndarray,
  language: Optional[np.ndarray] = None,
  **kwargs
  ) -> np.ndarray:
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

  number_groups = []
  for group_num, unique in enumerate(uniques):
    group_kwargs = {k: v[group_num] for k, v in array_groups.items()}
    number_group = pattern_functions[unique]['string_to_number'](
      **group_kwargs
    )
    number_groups.append(number_group)

  number = np.concatenate(number_groups)

  if len(group_indices[1:]):
    number = number[inverse_indices]

  return number




def get_cast_and_func(func):
  def _cast_and_func(number, *args, **kwargs):
    integer = np.array(number).astype(np.int64)

    return func(integer, *args, **kwargs)
  return _cast_and_func


def get_func_and_cast(func):
  def _func_and_cast(string, *args, **kwargs):
    integer = func(string, *args, **kwargs)
    number = np.array(integer).astype(np.float32)

    return number
  return _func_and_cast


def cast_number_to_string(
  number: np.ndarray,
  **kwargs
  ) -> np.ndarray:
  return number.astype(str)


def cast_string_to_number(
  string: np.ndarray,
  **kwargs
  ) -> np.ndarray:
  return string.astype(np.float32)

def dec_to_proper_frac(dec):
    sign = "-" if dec < 0 else ""
    abs_val = abs(dec)
    frac = fractions.Fraction(str(abs_val)).limit_denominator()
    return (f"{sign}{frac.numerator // frac.denominator} "
            f"{frac.numerator % frac.denominator}/{frac.denominator}")


def fraction_number_to_string(
  number: np.ndarray,
  whole: Optional[np.ndarray] = None,
  **kwargs
  ) -> np.ndarray:
  if whole is None:
    whole = np.array([False] * len(number))

  string = []
  for num, wh in zip(number, whole):
    if wh:
      s = dec_to_proper_frac(num).lstrip('0 ')
    else:
      s = str(fractions.Fraction(str(num)).limit_denominator())
    string.append(s)

  return np.array(string)


def fraction_string_to_number(
  string: np.ndarray,
  **kwargs
  ) -> np.ndarray:

  number = []
  for s in string:
    splits = s.split('/')
    if len(splits) != 2:
      raise ValueError("Malformed fraction {}".format(s))

    num_splits = splits[0].split()
    whole_number = '0'
    denominator = splits[1]

    if len(num_splits) == 1:
      numerator = num_splits[0]
    elif len(num_splits) == 2:
      whole_number = num_splits[0]
      numerator = num_splits[1]
    else:
      raise ValueError("Malformed fraction {}".format(s))

    number.append(float(whole_number) + float(numerator)/float(denominator))
  return np.array(number, dtype=np.float32)


pattern_functions = {
  'cast': {
    'number_to_string': cast_number_to_string,
    'string_to_number': cast_string_to_number},
  'fraction': {
    'number_to_string': fraction_number_to_string,
    'string_to_number': fraction_string_to_number},

}
for pattern in integer_dg.pattern_functions:
  new_key = 'integer_' + pattern

  pattern_functions[new_key] = {
    'number_to_string': get_cast_and_func(
      integer_dg.pattern_functions[pattern]['integer_to_string']),
    'string_to_number': get_func_and_cast(
      integer_dg.pattern_functions[pattern]['string_to_integer'])
  }
