from typing import Optional, List
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_transform.tf_metadata import schema_utils


def get_raw_feature_spec(schema: schema_pb2.Schema):
  """Get the feature spec from the schema."""
  return schema_utils.schema_as_feature_spec(schema).feature_spec
