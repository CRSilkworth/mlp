from typing import Optional, List, Dict, Text
from google.cloud import storage


import tensorflow as tf
import mlp.data_generation.integer as integer_dg
import mlp.data_generation.number as number_dg
import mlp.data_generation.date as date_dg
import mlp.data_generation.time as time_dg
import mlp.data_generation.maps as maps_dg
import pandas as pd
import numpy as np
import tempfile
import shutil
import os
import absl
import datetime

_MAX_INT = 9223372036854775807

class RandomDataGenerator:
  def __init__(
    self,
    random_value_parameters,
    batch_size: Optional[int] = 1,
    batches_per_file: Optional[int] = 10,
    keep_local_data: Optional[bool] = False,
    file_pattern: Optional[Text] = None,
    example_id_offset: Optional[int] = 0,
    file_offset: Optional[int] = 0,
    constant_columns: Optional[Dict[Text, str]] = None,
    data_type_probs: Optional[List[float]] = None,
    languages: Optional[List[str]] = None,
    max_tries: Optional[int] = 1,
    shuffle_frac: Optional[float] = 1.0,
    random_seed: Optional[int] = None,

    **kwargs
  ):
    self.random_value_parameters = random_value_parameters
    self.batch_size = batch_size
    self.random_seed = random_seed
    self.example_id_offset = example_id_offset
    self.file_offset = file_offset
    self.keep_local_data = keep_local_data
    self.file_pattern = file_pattern or ''
    self.batches_per_file = batches_per_file
    self.constant_columns = constant_columns or {}
    self.data_type_probs = data_type_probs or [1./len(self.random_value_parameters)] * len(self.random_value_parameters)
    self.languages = languages or ['en', 'ja', 'zh_hans', 'zh_hant', 'ko']
    self.max_tries = max_tries
    self.shuffle_frac = shuffle_frac

    if self.random_seed is not None:
      np.random.seed(self.random_seed)

  def __enter__(self):
    self.tempdir = tempfile.mkdtemp()

    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    shutil.rmtree(self.tempdir)

  def get_random_draw(self, parameters, batch_size):
    random_draw = {}

    if 'categorical' in parameters:
      for param_key in parameters['categorical']:
        allowed_values = parameters['categorical'][param_key]
        random_draw[param_key] = np.random.choice(allowed_values, batch_size)
    if 'bounded' in parameters:
      for param_key in parameters['bounded']:
        upper_bound = parameters['bounded'][param_key]
        random_draw[param_key] = np.random.randint(upper_bound, size=batch_size)
    if 'normal' in parameters:
      for param_key in parameters['normal']:
        mu, sigma = parameters['normal'][param_key]
        random_draw[param_key] = np.random.normal(mu, sigma, batch_size)

    if 'integer' in parameters:
      for param_key in parameters['integer']:
        random_draw[param_key] = self._get_random_integer(batch_size)

    if 'number' in parameters:
      for param_key in parameters['number']:
        random_draw[param_key] = self._get_random_number(batch_size)

    return random_draw

  def _correct_integer_data(self, random_draw):
    return random_draw

  def _correct_number_data(self, random_draw):
    iterator = date_dg.zip_many(
      **random_draw
    )

    for r_num, r in enumerate(iterator):
      if r['pattern'].startswith('integer'):
        random_draw['number'][r_num] = float(int(r['number']))

    return random_draw

  def _correct_date_data(self, random_draw):

    random_draw = self._convert_day_to_date(random_draw)
    random_draw = self._convert_day_to_date(random_draw, prefix='cur_')

    cur_date = date_dg.tuple_to_datetime(random_draw['cur_year'], random_draw['cur_month'], random_draw['cur_day'])
    date = date_dg.tuple_to_datetime(random_draw['year'], random_draw['month'], random_draw['day'])

    cur_day_of_week = pd.Series(cur_date).dt.dayofweek
    day_of_week = pd.Series(date).dt.dayofweek

    iterator = date_dg.zip_many(
      cur_date=cur_date,
      cur_day_of_week=cur_day_of_week,
      date=date,
      day_of_week=day_of_week,
      **random_draw
    )

    dates = []
    for r_num, r in enumerate(iterator):
      if r['pattern'] == 'ja':
        random_draw['language'][r_num] = 'ja'

      if r['pattern'] == 'ko':
        random_draw['language'][r_num] = 'ko'
      if r['pattern'] in ('d_word_y', 'word_dy'):
        if r['postfix']:
          random_draw['language'][r_num] = 'en'

      if r['pattern'] == 'delta_day_of_week':
        nums = len(date_dg.next_words[r['next']][r['language']])
        if r['num'] >= nums:
          r['num'] = np.random.randint(nums)
          random_draw['num'][r_num] = r['num']
        day_diff = (r['day_of_week'] - r['cur_day_of_week']) % 7
        if r['next'] == 'this':
          day_diff = day_diff if day_diff > 0 else 7
        elif r['next'] == 'next':
          day_diff = day_diff + 7
        elif r['next'] == 'last':
          day_diff = -1 * ((r['cur_day_of_week'] - r['day_of_week']) % 7)

        date = r['cur_date'] + day_diff
        dates.append(date)

      elif r['pattern'] == 'delta_day':
        if r['num_days'] not in date_dg.delta_day_inv[r['language']]:
          r['num_days'] = np.random.choice(date_dg.delta_day_inv[r['language']].keys())
          random_draw['num'][r_num] = r['num']

        nums = len(date_dg.delta_day_inv[r['language']][r['num_days']])
        if r['num'] >= nums:
          r['num'] = np.random.randint(nums)
        random_draw['num'][r_num] = r['num']

        date = r['cur_date'] + r['num_days']
        dates.append(date)
      else:
        dates.append(r['date'])

    random_draw['year'], random_draw['month'], random_draw['day'] = date_dg.date_to_tuple(np.array(dates))

    return random_draw

  def _correct_time_data(self, random_draw):
    iterator = date_dg.zip_many(
      **random_draw
    )

    for r_num, r in enumerate(iterator):
      am_pm_type = np.random.choice(list(maps_dg.am_pm_to_tuple[r['language']].keys()))
      random_draw['am_pm_type'][r_num] = am_pm_type

      if maps_dg.am_pm_to_tuple[r['language']][am_pm_type][0] > 12 and r['hour'] < 12:
        random_draw['hour'][r_num] = random_draw['hour'][r_num] + 12
      elif maps_dg.am_pm_to_tuple[r['language']][am_pm_type][0] < 12 and r['hour'] > 12:
        random_draw['hour'][r_num] = random_draw['hour'][r_num] - 12

      if r['pattern'] == 'digit_hour':
        random_draw['minute'][r_num] = 0
        if r['hour_word'] == ':':
          random_draw['hour_word'][r_num] = ''

      if r['short_cut_word'] not in maps_dg.short_cut_words[r['language']]:
        r['short_cut_word'] = np.random.choice(list(maps_dg.short_cut_words[r['language']].keys()))

        r['minute'] = maps_dg.short_cut_words[r['language']][r['short_cut_word']][1]
        random_draw['short_cut_word'][r_num] = r['short_cut_word']

      if r['short_cut_word'] != '':
        random_draw['minute'][r_num] = maps_dg.short_cut_words[r['language']][r['short_cut_word']][1]

      if r['hour_word'] not in maps_dg.hour_words[r['language']]:
        r['hour_word'] = np.random.choice(list(maps_dg.hour_words[r['language']]))

        random_draw['hour_word'][r_num] = r['hour_word']

      if r['minute_word'] not in maps_dg.minute_words[r['language']]:
        r['minute_word'] = np.random.choice(list(maps_dg.minute_words[r['language']]))

        random_draw['minute_word'][r_num] = r['minute_word']

      if r['digit_type'] not in maps_dg.integer_to_string[r['language']][0]:
        r['digit_type'] = np.random.choice(
          list(maps_dg.integer_to_string[r['language']][0].keys())
        )

        random_draw['digit_type'][r_num] = r['digit_type']

    return random_draw

  def write_data(self, num_batches, bucket_name, cols_to_keep):
    inputs = {}
    example_id_offset = self.example_id_offset
    num_files = num_batches // self.batches_per_file
    for file_num in range(num_files):
      dfs = []
      for batch_num in range(self.batches_per_file):
        example_id_offset += self.batch_size
        if batch_num % 100 == 0:
          print('-'*100)
          print(file_num, batch_num, example_id_offset - self.example_id_offset)
          print('-'*100)
        df = self.generate_batch(file_num, batch_num, example_id_offset=example_id_offset)
        dfs.append(df)

      dfs = pd.concat(dfs)
      local_file_name = os.path.join(
        self.tempdir,
        self.file_pattern, str(file_num + self.file_offset).zfill(5) + '.csv'
      )
      os.makedirs(os.path.join(self.tempdir, self.file_pattern), exist_ok=True)

      gcp_file_name = os.path.join(
        self.file_pattern, str(file_num + self.file_offset).zfill(5) + '.csv'
      )

      dfs.set_index('example_id', inplace=True)
      dfs = dfs[cols_to_keep]
      dfs.to_csv(local_file_name, encoding='utf-8', index_label='example_id', date_format='%Y-%m-%d %H:%M:%S')

      upload_to_bucket(bucket_name, gcp_file_name, local_file_name)

  def generate_batch(self, file_num, batch_num, example_id_offset=0):
    start_id = example_id_offset
    end_id = start_id + self.batch_size
    example_ids = list(range(start_id, end_id))
    df = pd.DataFrame({'example_id': example_ids})

    for column in self.constant_columns:
      df[column] = self.constant_columns[column]

    df['data_type_id'] = np.random.choice(list(self.random_value_parameters.keys()), self.batch_size, p=self.data_type_probs)

    generated_integer_df = self.generate_integer(
      df[df['data_type_id'] == 0], self.random_value_parameters[0])
    generated_number_df = self.generate_number(
      df[df['data_type_id'] == 1], self.random_value_parameters[1])
    generated_date_df = self.generate_date(
      df[df['data_type_id'] == 2], self.random_value_parameters[2])
    generated_time_df = self.generate_time(
      df[df['data_type_id'] == 3], self.random_value_parameters[3])

    df = pd.concat([generated_integer_df, generated_number_df, generated_date_df, generated_time_df])

    if self.shuffle_frac is not None:
      df = df.sample(frac=self.shuffle_frac).reset_index(drop=True)

    return df

  def generate_number(self, df, random_value_parameters):
    df = df.copy(deep=True)
    num_rows = len(df)
    if not len(df):
      df['span'] = []
      df['float'] = []
      return df

    num_tries = 0
    failed = True
    spans = []
    while failed:
      random_draw = self.get_random_draw(random_value_parameters, num_rows)
      random_draw = self._correct_number_data(random_draw)
      try:
        span = number_dg.number_to_string(**random_draw)
        failed = False
        spans.append(span)

      except Exception as e:
        num_tries += 1

        if num_tries > self.max_tries:
          raise e

    df['span'] = np.concatenate(spans, axis=0)
    df['float'] = random_draw['number']
    df['user_language'] = random_draw['language']
    df['pattern'] = random_draw['pattern']
    return df

  def generate_integer(self, df, random_value_parameters):
    df = df.copy(deep=True)
    num_rows = len(df)
    if not len(df):
      df['span'] = []
      df['integer'] = []
      return df

    num_tries = 0
    failed = True
    spans = []
    while failed:
      random_draw = self.get_random_draw(random_value_parameters, num_rows)
      random_draw = self._correct_integer_data(random_draw)
      try:
        span = integer_dg.integer_to_string(**random_draw)
        failed = False
        spans.append(span)

      except Exception as e:
        num_tries += 1

        if num_tries > self.max_tries:
          raise e

    df['span'] = np.concatenate(spans, axis=0)
    df['integer'] = random_draw['integer']
    df['user_language'] = random_draw['language']
    df['pattern'] = random_draw['pattern']
    return df

  def _convert_day_to_date(self, random_draw, prefix=''):

    year = []
    month = []
    day = []

    for delta in random_draw[prefix + 'delta_day']:
      date = datetime.datetime(1970, 1, 1) + datetime.timedelta(delta)
      year.append(date.year)
      month.append(date.month)
      day.append(date.day)

    random_draw[prefix + 'year'] = np.array(year)
    random_draw[prefix + 'month'] = np.array(month)
    random_draw[prefix + 'day'] = np.array(day)

    return random_draw

  def generate_date(self, df, random_value_parameters):
    df = df.copy(deep=True)
    num_rows = len(df)
    if not len(df):
      df['span'] = []
      df['year'] = []
      df['month'] = []
      df['day'] = []
      df['cur_year'] = []
      df['cur_month'] = []
      df['cur_day'] = []
      return df

    num_tries = 0
    failed = True
    spans = []
    while failed:
      random_draw = self.get_random_draw(random_value_parameters, num_rows)
      random_draw = self._correct_date_data(random_draw)

      try:
        span = date_dg.date_to_string(**random_draw)
        failed = False
        spans.append(span)

      except Exception as e:
        num_tries += 1

        if num_tries > self.max_tries:
          raise e

    df['span'] = np.concatenate(spans, axis=0)
    df['year'] = random_draw['year']
    df['month'] = random_draw['month']
    df['day'] = random_draw['day']
    df['cur_year'] = random_draw['cur_year']
    df['cur_month'] = random_draw['cur_month']
    df['cur_day'] = random_draw['cur_day']
    df['user_language'] = random_draw['language']
    df['pattern'] = random_draw['pattern']

    return df

  def generate_time(self, df, random_value_parameters):
    df = df.copy(deep=True)
    num_rows = len(df)
    if not len(df):
      df['span'] = []
      df['hour'] = []
      df['minute'] = []
      df['second'] = []
      return df

    num_tries = 0
    failed = True
    spans = []
    while failed:
      random_draw = self.get_random_draw(random_value_parameters, num_rows)
      random_draw = self._correct_time_data(random_draw)
      try:
        span = time_dg.time_to_string(**random_draw)
        failed = False
        spans.append(span)
      except Exception as e:
        num_tries += 1

        if num_tries > self.max_tries:
          raise e

    df['span'] = np.concatenate(spans, axis=0)
    df['hour'] = random_draw['hour']
    df['minute'] = random_draw['minute']
    df['second'] = random_draw['second']
    df['user_language'] = random_draw['language']
    df['pattern'] = random_draw['pattern']

    return df

  def _get_random_integer(self, num, decay=2.0):
    primitives = np.array(
      [None] + list(range(20)) + [10**i for i in range(2, 10)]
    )
    probs = [1.] * 20
    probs += [1./(decay * ind + 1) for ind in range(len(primitives[20:]))]
    probs = np.array(probs)
    probs = probs / np.sum(probs)

    integers = []
    for integer_num in range(num):
      draw = np.random.choice(primitives[1:], p=probs[1:]/(1. - probs[0]))
      integer = draw
      while True:
        prev_integer = integer
        draw = np.random.choice(primitives, p=probs)

        if draw == 0:
          continue
        elif draw is None:
          break

        multiply = np.random.choice([True, False])
        if multiply:
          integer = integer * draw
        else:
          integer = integer + draw

        if integer >= _MAX_INT:
          integer = prev_integer
          break

      integers.append(prev_integer)
    return np.array(integers)

  def _get_random_number(self, num, decay=2.0):

    numerator = self._get_random_integer(num, decay)
    denominator = self._get_random_integer(num, decay)

    while (denominator == 0).any():
      zero = denominator == 0

      denominator[zero] = self._get_random_integer(len(denominator[zero]), decay)

    return numerator / denominator


def upload_to_bucket(bucket_name, blob_path, local_path):
    bucket = storage.Client().bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
