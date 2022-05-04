from absl import logging
from subprocess import Popen, PIPE


def build_image(image_name, image_tag, dir='.', docker_file=None):
  """Build docker image."""
  if docker_file is None:
    command = 'docker build . -t {image_name}:{image_tag}'.format(image_name=image_name, image_tag=image_tag)
  else:
    command = 'docker build . -f {docker_file} -t {image_name}:{image_tag}'.format(image_name=image_name, image_tag=image_tag, docker_file=docker_file)

  p = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)

  stdout, stderr = p.communicate()

  logging.info('build_image stdout:')
  logging.info(stderr)
  logging.error('build_image stderr:')
  logging.error(stdout)
  return p.returncode


def push_image(image_name, image_tag):
  """Push docker image."""
  command = 'docker push {image_name}:{image_tag}'.format(image_name=image_name, image_tag=image_tag)
  p = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)

  stdout, stderr = p.communicate()

  logging.info('push_image stdout:')
  logging.info(stdout)
  logging.error('push_image stderr:')
  logging.error(stderr)

  return p.returncode


def local_image_exists(image_name, image_tag):
  """Check if image exists locally."""
  command = 'docker inspect --type=image {image_name}:{image_tag} > /dev/null; echo $?'.format(image_name=image_name, image_tag=image_tag)
  p = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)

  stdout, stderr = p.communicate()
  return int(stdout.strip()) == 0


def remote_image_exists(image_name, image_tag):
  """Check if image exists on GCP."""
  command = 'docker manifest inspect {image_name}:{image_tag} > /dev/null ; echo $?'.format(image_name=image_name, image_tag=image_tag)
  p = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)

  stdout, stderr = p.communicate()

  return int(stdout.strip()) == 0
