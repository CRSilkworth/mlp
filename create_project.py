from __future__ import division
from __future__ import print_function
from typing import List, Any
from absl import flags
from absl import app
from absl import logging

import os
import shutil

FLAGS = flags.FLAGS
flags.DEFINE_string('mlp_project', None, 'The name of the project. Will be the root directory of the project.')
flags.DEFINE_string('mlp_subproject', None, 'The name of the subproject. Will be a sub directory inside the project directory.')
flags.DEFINE_string('gcp_project', None, 'The name of the GCP project. Any GCP services will be used with the account corresponding to this project. (If using GCP Services)')
flags.DEFINE_string('gcp_bucket', None, 'The name of the GCP bucket to store the files generated by running a pipeline. (If using GCP Services)')
flags.DEFINE_string('gcp_zone', 'asia-east1-b', 'The name of the GCP zone to run kubeflow, dataflow, etc. from. (If using GCP Services)')
flags.DEFINE_string('gcp_credentials_json', '', 'Path to your GOOGLE_APPLICATION_CREDENTIALS json file. (If using GCP Services)')
flags.DEFINE_string('iap_client_id', '', 'The oauth client id tied to your kubeflow deployment. (If using running from Kubeflow on GCP)')
flags.DEFINE_string('iap_client_secret', '', 'The oauth client secret associated with your client id. (If using running from Kubeflow on GCP)')

flags.DEFINE_string('dir', './', 'The directory to place the example project.')
flags.mark_flag_as_required('mlp_project')
flags.mark_flag_as_required('mlp_subproject')
flags.mark_flag_as_required('gcp_project')
flags.mark_flag_as_required('gcp_bucket')

def replace_strings_in_dir(dir, string_map):
  for dir_name, dirs, files in os.walk(dir):
    for file_name in files:
      file_path = os.path.join(dir_name, file_name)

      try:
        with open(file_path) as f:
          text = f.read()
      except UnicodeDecodeError:
        continue

      for old_string in string_map:
        text = text.replace(old_string, string_map[old_string])

      with open(file_path, "w") as f:
        f.write(text)


def main(argv: List[Any]):
  os.makedirs(FLAGS.dir, exist_ok=True)

  string_map = {
    '__gcp_project__': FLAGS.gcp_project,
    '__gcp_bucket__': FLAGS.gcp_bucket,
    '__example_project__': FLAGS.mlp_project,
    '__example_subproject__': FLAGS.mlp_subproject,
    '__gcp_zone__': FLAGS.gcp_zone,
    '__gcp_region__': FLAGS.gcp_zone[:-2],
    '__gcp_credentials_json__': FLAGS.gcp_credentials_json,
    '__iap_client_id__': FLAGS.iap_client_id,
    '__iap_client_secret__': FLAGS.iap_client_secret
  }

  example_project_dir = os.path.join(os.path.dirname(__file__), 'example_project')
  example_subproject_dir = os.path.join(os.path.dirname(__file__), 'example_project', 'example_subproject')

  subproject_dir = os.path.join(FLAGS.dir, FLAGS.mlp_project, FLAGS.mlp_subproject)

  project_dir = os.path.join(FLAGS.dir, FLAGS.mlp_project)
  subproject_dir = os.path.join(FLAGS.dir, FLAGS.mlp_project, FLAGS.mlp_subproject)

  if os.path.exists(project_dir):
    logging.info('{} already exists. Will not alter any files outside of {}'.format(project_dir, subproject_dir))

    if os.path.exists(subproject_dir):
      logging.info('{} already exists. Nothing to be done. Exiting'.format(subproject_dir))
      return

    shutil.copytree(example_subproject_dir, subproject_dir)
    replace_strings_in_dir(subproject_dir, string_map)
  else:
    print(example_project_dir, project_dir)
    shutil.copytree(example_project_dir, project_dir)

    os.rename(
      os.path.join(project_dir, 'example_subproject'), subproject_dir
    )
    os.rename(
      os.path.join(project_dir, 'set_env_template.sh'),
      os.path.join(project_dir, 'set_env.sh')
    )
    replace_strings_in_dir(project_dir, string_map)

if __name__ == "__main__":
  app.run(main)
