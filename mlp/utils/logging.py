import absl


def capture_all_exceptions(func):
  def wrapper_func(*args, **kwargs):
    absl.logging.get_absl_handler().use_absl_log_file()
    try:
      func(*args, **kwargs)
    except Exception as e:
      absl.logging.exception(e)
      raise e
  return wrapper_func
