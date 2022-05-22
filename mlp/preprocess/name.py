from typing import Optional, Text, List


def trans_name(key: Text, postfix: Optional[Text] = '_xf') -> Text:
  """Transform a feature key to it's transform's feature key."""
  return key + postfix


def trans_names(keys: List[Text], postfix: Optional[Text] = '_xf') -> List[Text]:
  """Transform a list of feature keys to their transforms' feature keys."""
  return [trans_name(key, postfix) for key in keys]
