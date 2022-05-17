import os
import shutil
import tempfile
import tensorflow as tf
from absl import flags
from typing import List, Text, Any
from absl import logging
from absl import app
from mlp.utils.resolvers import latest_artifact_path
from mlp.utils.dirs import copy_dir

FLAGS = flags.FLAGS

flags.DEFINE_string('pipeline_run_dir', None, 'The directory where all the pipeline data is stored.')
flags.DEFINE_string('copy_destination', '.', 'Where to copy the model to.')

flags.mark_flag_as_required('pipeline_run_dir')


def main(argv: List[Any]):
  pipeline_uri = os.path.join(FLAGS.pipeline_run_dir)
  copy_model(pipeline_uri)


def copy_model(pipeline_uri: Text):
  model_uri = latest_artifact_path(pipeline_uri, 'data/model_pusher/pushed_model')
  config_file_name = 'pipeline_vars.json'
  config_uri = os.path.join(pipeline_uri, 'config', config_file_name)
  try:
    model_time_stamp = int(model_uri.split('/')[-1])
  except ValueError:
    raise ValueError("Did not get a valid integer time stamp as last directory in latest model uri {}.".format(model_uri))

  logging.info(("Copying contents of {} to {}".format(model_uri, FLAGS.copy_destination)))
  with tempfile.TemporaryDirectory() as temp_dir:
    temp_path = os.path.join(temp_dir, str(model_time_stamp))
    tf.io.gfile.makedirs(temp_path)

    copy_dir(model_uri.rstrip('/'), temp_path, ignore_subdirs='checkpoints')
    tf.io.gfile.copy(config_uri, os.path.join(temp_path, config_file_name))

    shutil.move(os.path.join(temp_dir, str(model_time_stamp)), FLAGS.copy_destination)


if __name__ == "__main__":
  app.run(main)
