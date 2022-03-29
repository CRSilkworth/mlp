import collections
from typing import Optional, List, Text, Dict
from tensorflow_metadata.proto.v0 import schema_pb2
from tfx.utils import io_utils

full_schema = collections.OrderedDict([
  ('vendor', 'STRING'),
  ('max_bottle_volume', 'FLOAT64'),
  ('category', 'STRING'),
])


def write_default_schema(
  schema_uri: Text,
  input_keys: List[Text],
  label_keys: List[Text],
  ):
  """
  Write a default schema based off the inputs and labels provided.

  Parameters
  ----------
  schema_uri: The path to where the schema should be saved.
  input_keys: The list of all input keys
  label_keys: The list of all label keys

  """
  string_keys = []
  int_keys = []
  float_keys = []

  # Split up the keys by their data type.
  for key_set in [input_keys, label_keys]:
      for key in key_set:
        if full_schema[key] == 'STRING':
          string_keys.append(key)
        elif full_schema[key] == 'INT64':
          int_keys.append(key)
        elif full_schema[key] == 'FLOAT64':
          float_keys.append(key)

  # Generate the schema
  schema = generate_default_schema(
    string_keys=string_keys,
    int_keys=int_keys,
    float_keys=float_keys
  )

  # Write it to the disk
  io_utils.write_pbtxt_file(schema_uri, schema)


def generate_default_schema(
  string_keys: List[Text],
  int_keys: List[Text],
  float_keys: List[Text]
  ) -> schema_pb2.Schema:
  """
  Generate a schema protobuff from lists of keys.

  Parameters
  ----------
  string_keys: The keys with string type.
  integer_keys: The keys with integer type.
  float_keys: The keys with float type.

  Returns
  -------
  schema: A schema protobuff

  """
  schema = schema_pb2.Schema()

  for key in string_keys:
    schema.feature.add()
    feature = schema.feature[-1]
    feature.name = key
    feature.type = 1
    feature.value_count.min = 1
    feature.value_count.max = 1
    feature.presence.min_count = 1

  for key in int_keys:
    schema.feature.add()
    feature = schema.feature[-1]
    feature.name = key
    feature.type = 2
    feature.value_count.min = 1
    feature.value_count.max = 1
    feature.presence.min_count = 1

  for key in float_keys:
    schema.feature.add()
    feature = schema.feature[-1]
    feature.name = key
    feature.type = 3
    feature.value_count.min = 1
    feature.value_count.max = 1
    feature.presence.min_count = 1

  return schema
