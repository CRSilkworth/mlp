from absl import logging
import contextlib
import sys


def capture_all_exceptions(func, level=logging.INFO):
  def wrapper_func(*args, **kwargs):
    logging.get_absl_handler().use_absl_log_file()
    logging.set_verbosity(level)
    try:
      func(*args, **kwargs)
    except Exception as e:
      logging.exception(e)
      raise e
  return wrapper_func
