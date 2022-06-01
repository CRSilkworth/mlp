class Obj(dict):
  __getattr__ = dict.get
