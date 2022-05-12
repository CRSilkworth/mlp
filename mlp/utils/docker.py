"""Simple helper functions for dealing with docker images."""
from absl import logging
from subprocess import Popen, PIPE
from typing import Text, Optional


def build_image(
  image_name: Text,
  image_tag: Text,
  dir: Optional[Text] = '.',
  docker_file: Optional[Text] = None) -> int:
  """Build docker image."""
  if docker_file is None:
    command = 'docker build . -t {image_name}:{image_tag}'.format(image_name=image_name, image_tag=image_tag)
  else:
    command = 'docker build . -f {docker_file} -t {image_name}:{image_tag}'.format(image_name=image_name, image_tag=image_tag, docker_file=docker_file)

  p = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)

  stdout, stderr = p.communicate()

  logging.info('build_image stdout:')
  logging.info(stderr.decode('utf-8'))
  logging.error('build_image stderr:')
  logging.error(stdout.decode('utf-8'))
  return p.returncode


def push_image(
  image_name: Text,
  image_tag: Text) -> int:
  """Push docker image."""
  command = 'docker push {image_name}:{image_tag}'.format(image_name=image_name, image_tag=image_tag)
  p = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)

  stdout, stderr = p.communicate()

  logging.info('push_image stdout:')
  logging.info(stdout.decode('utf-8'))
  logging.error('push_image stderr:')
  logging.error(stderr.decode('utf-8'))

  return p.returncode


def pull_image(
  image_name: Text,
  image_tag: Text) -> int:
  """Push docker image."""
  command = 'docker pull {image_name}:{image_tag}'.format(image_name=image_name, image_tag=image_tag)
  p = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)

  stdout, stderr = p.communicate()

  logging.info('push_image stdout:')
  logging.info(stdout.decode('utf-8'))
  logging.error('push_image stderr:')
  logging.error(stderr.decode('utf-8'))

  return p.returncode


def local_image_exists(
  image_name: Text,
  image_tag: Text) -> bool:
  """Check if image exists locally."""
  command = 'docker inspect --type=image {image_name}:{image_tag} > /dev/null; echo $?'.format(image_name=image_name, image_tag=image_tag)
  p = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)

  stdout, stderr = p.communicate()
  return int(stdout.strip()) == 0


def remote_image_exists(
  image_name: Text,
  image_tag: Text) -> bool:
  """Check if image exists on GCP."""
  command = 'docker manifest inspect {image_name}:{image_tag} > /dev/null ; echo $?'.format(image_name=image_name, image_tag=image_tag)
  p = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)

  stdout, stderr = p.communicate()

  
  return int(stdout.strip()) == 0
