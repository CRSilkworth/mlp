"""Package Setup script for mlp."""

from __future__ import print_function

from setuptools import find_packages
from setuptools import setup

import proto_gen
import glob

for file_pattern in proto_gen._PROTO_FILE_PATTERNS:
  for proto_file in glob.glob(file_pattern):
    proto_gen.generate_proto(proto_file)

with open('requirements.txt') as fp:
  requirements = []
  for line in fp:
    requirements.append(line)


def _make_required_install_packages():
  return requirements


# Get version from version module.
with open('version.py') as fp:
  globals_dict = {}
  exec(fp.read(), globals_dict)  # pylint: disable=exec-used
__version__ = globals_dict['__version__']

# Get the long description from the README file.
with open('README.md') as fp:
  _LONG_DESCRIPTION = fp.read().format(version=__version__)

# package_dir = {
#   'mlp.' + p: p for p in find_packages()
# }
# packages = ['mlp.' + p for p in find_packages()]
# print(package_dir)
# print(packages)
setup(
    name='mlp',
    version=__version__,
    author='Christopher Silkworth',
    author_email='',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Operating System :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    namespace_packages=[],
    install_requires=_make_required_install_packages(),
    python_requires='>2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*,<4',
    # package_dir=package_dir,
    packages=['mlp.' + p for p in find_packages(where='./mlp')],
    include_package_data=True,
    description='End to end machine learning pipelines for automated data preprocessing, anomaly detection, training and deployment.',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    requires=[]
    )
