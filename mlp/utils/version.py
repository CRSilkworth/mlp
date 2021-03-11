import importlib

def get_project_version(version_file_path='./version.py'):
  spec = importlib.util.spec_from_file_location("version", version_file_path)
  version = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(version)

  return version.__version__
