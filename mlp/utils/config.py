import tensorflow as tf
import tempfile
import os
import json
from hashlib import sha1


class VarConfig(object):
  def __init__(self, file_path=None):
    self.file_path = file_path
    self.vars = {}

    self.is_initialized = True
    if file_path is not None and tf.io.gfile.exists(file_path):
      self._load_from_json(file_path)
    elif file_path is not None:
      raise ValueError("{} not found".format(file_path))

  def _load_from_json(self, file_path):
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_file_name = os.path.join(temp_dir, 'temp.py')
      tf.io.gfile.copy(file_path, temp_file_name)
      with open(temp_file_name, 'r') as temp_file:
        vars = json.load(temp_file)
    self.add_vars(**vars)

  def __setattr__(self, name, value):
    if hasattr(self, 'is_initialized') and self.is_initialized:
      self.vars[name] = value
      super(VarConfig, self).__setattr__(name, value)
    else:
      super(VarConfig, self).__setattr__(name, value)

  def get_hash(self):
    temp_d = {k: v for k, v in self.var_names.items() if k != 'hash'}

    hash = sha1(json.dumps(temp_d, sort_keys=True))

  def get_vars(self):
    r_dict = {}
    r_dict.update(self.vars)
    return r_dict

  def write(self, file_path=None, overwrite=False):
    if file_path is None:
      raise ValueError("file_path is not set")

    if tf.io.gfile.exists(file_path) and overwrite is True:
      raise ValueError("{} exists and overwrite is set to False".format(file_path))

    with tempfile.TemporaryDirectory() as temp_dir:
      temp_file_name = os.path.join(temp_dir, 'temp.py')

      with open(temp_file_name, 'w') as temp_file:
        json.dump(self.vars, temp_file, indent=2, sort_keys=True)

      save_dir = '/'.join(file_path.split('/')[:-1])
      if not tf.io.gfile.exists(save_dir):
        tf.io.gfile.makedirs(save_dir)

      tf.io.gfile.copy(temp_file_name, file_path)

  def add_vars(self, **kwargs):
    for key, val in kwargs.items():
      setattr(self, key, val)
