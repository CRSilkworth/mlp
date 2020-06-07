import os
import sys
import subprocess
from distutils.spawn import find_executable


def generate_proto(source):
  """Invokes the Protocol Compiler to generate a _pb2.py."""

  # Find the Protocol Compiler.
  if 'PROTOC' in os.environ and os.path.exists(os.environ['PROTOC']):
    protoc = os.environ['PROTOC']
  elif os.path.exists('../src/protoc'):
    protoc = '../src/protoc'
  elif os.path.exists('../src/protoc.exe'):
    protoc = '../src/protoc.exe'
  elif os.path.exists('../vsprojects/Debug/protoc.exe'):
    protoc = '../vsprojects/Debug/protoc.exe'
  elif os.path.exists('../vsprojects/Release/protoc.exe'):
    protoc = '../vsprojects/Release/protoc.exe'
  else:
    protoc = find_executable('protoc')

  output = source.replace('.proto', '_pb2.py')

  if (not os.path.exists(output) or
      (os.path.exists(source) and
       os.path.getmtime(source) > os.path.getmtime(output))):
    print('Generating %s...' % output)

    if not os.path.exists(source):
      sys.stderr.write('Cannot find required file: %s\n' % source)
      sys.exit(-1)

    if protoc is None:
      sys.stderr.write(
          'protoc is not installed nor found in ../src.  Please compile it '
          'or install the binary package.\n')
      sys.exit(-1)

    protoc_command = [protoc, '-I.', '--python_out=.', source]
    if subprocess.call(protoc_command) != 0:
      sys.exit(-1)


if __name__ == "__main__":
  source = sys.argv[1]
  generate_proto(source)
