"""Convience script to launch a run to kubeflow. Handles the experiment, pipeline, pipeline version, and run creation as well as building an image if necessary."""
from absl import flags
from absl import app
from typing import Optional, Text, Any, List

from mlp.scripts.create_update_run import create_update_run

FLAGS = flags.FLAGS
flags.DEFINE_string('pipeline_path', None, 'The path to the pipeline definition file.')
flags.DEFINE_string('experiment', 'dev', 'The experiment to run the pipeline under. Defaults to "dev"')

flags.mark_flag_as_required('pipeline_path')


def main(argv):
  create_update_run(FLAGS.pipeline_path, FLAGS.experiment)

if __name__ == "__main__":
  app.run(main)
