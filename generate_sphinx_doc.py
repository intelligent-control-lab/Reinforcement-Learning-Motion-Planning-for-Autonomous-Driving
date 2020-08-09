import os
import ast
import fnmatch
import shutil
from pathlib import Path
import collections
from src.utils import path_exists

files_to_ignore = ['__init__.py']
dirs_to_ignore = ['__pycache__']
valid_ext = ['.py', '.md']

'''Helper function for filtering out bad files
'''
def filter_files(list_of_files):
  for bad in files_to_ignore:
    list_of_files = [f for f in list_of_files if not fnmatch.fnmatch(f, bad)]

  list_of_files = [f for f in list_of_files if f.endswith(tuple(valid_ext))]
  return list_of_files

'''Helper function for filtering out bad directories
'''
def filter_dirs(list_of_dirs):
  for bad in dirs_to_ignore:
    list_of_dirs = [d for d in list_of_dirs if not fnmatch.fnmatch(d, bad)]
  return list_of_dirs

def check_directory(dir_):
  files = os.listdir(dir_)
  files = filter_files(files)
  return len(files) > 0

def create_sub_rst_file(args, base):
  sub_rst_file = os.path.join(args['doc_base_dir'], base + '.rst')

  parent, fn = os.path.split(sub_rst_file)
  path_exists(parent)

  sub_rst_f = open(sub_rst_file, 'w')
  sub_rst_f.write(f':mod:`{base}`\n'+'='*50+'\n\n')

  module = '.'.join(base.split('/'))
  sub_rst_f.write(f'.. automodule:: {module}\n.. toctree::\n\n')

  base_parent = str(Path(base).parent)

  for root, dirs, files in os.walk(base):
    files = filter_files(files)
    dirs = filter_dirs(dirs)

    for file in files:
      file = os.path.join(root, file)
      for ext in valid_ext:
        file = file.replace(ext, '')
      path = file.replace(base_parent + '/', '')
      sub_rst_f.write(f'    {path}\n')

    for d in dirs:
      path = os.path.join(root, d)
      if not check_directory(path): continue
      path = path.replace(base_parent + '/', '')
      sub_rst_f.write(f'    {path}\n')
    break

  sub_rst_f.write('\n')
  return sub_rst_f

def parse_source_file(file):
  with open(file) as f:
    node = ast.parse(f.read())

  functions = [n for n in node.body if isinstance(n, ast.FunctionDef)]
  classes = [n for n in node.body if isinstance(n, ast.ClassDef)]
  return functions, classes

def create_source_rst_file(file, base_f):
  # Parse the source file
  functions, classes = parse_source_file(file)

  base, filename = os.path.split(file)
  (header, ext) = os.path.splitext(filename)

  path = file.replace('.py', '')
  module = '.'.join(path.split('/'))

  try:
    exec(f'import {module}')
  except:
    print(f'Could not import: {file}')
    # import ipdb; ipdb.set_trace()
    # raise RuntimeError(f'Could not import: {file}')

  rst_file = os.path.join(args['doc_base_dir'], file.replace(ext, '.rst'))
  root, _ = os.path.split(rst_file)
  path_exists(root)

  print(f'Writing {rst_file}')

  with open(rst_file, 'w') as rst:
    # Write header
    rst.write(f'{header} \n'+'-'*50+'\n\n')

    print(f'Module: {module}')
    rst.write(f'.. automodule:: {module}\n\n')
    rst.write(f'.. currentmodule:: {module}\n\n')

    for function in functions:
      print("\t Function: ", function.name)
      base_f.write(f'    {function.name}\n')
      rst.write(f'.. autofunction:: {function.name}\n\n')


    for class_ in classes:
      print("\t Class name:", class_.name)
      base_f.write(f'    {class_.name}\n')
      rst.write(f'.. autoclass:: {class_.name}\n')
      rst.write(f'\t :members:\n\n')

  return rst_file

def create_markdown_rst_file(file, base_f):
  base, filename = os.path.split(file)
  (header, ext) = os.path.splitext(filename)

  path = file.replace('.md', '')

  rst_file = os.path.join(args['doc_base_dir'], file.replace(ext, '.rst'))
  root, _ = os.path.split(rst_file)
  path_exists(root)

  print(f'Writing {rst_file}')

  depth = len(file.split('/')) - 1

  with open(rst_file, 'w') as rst:
    extra = '../'*depth
    rst.write(f"{file}\n"+'='*50+'\n\n')
    rst.write(f".. mdinclude:: {extra}../../domain-adaptation-rl/{file}")

  return rst_file

'''Function for generating documentation for a single directory
'''
def generate_doc_for_dir(args, directory):
  # Mapping from {base_path: [source_files]}
  source_files = collections.defaultdict(list)
  for root, dirs, files in os.walk(directory):
    files = filter_files(files)

    for file in files:
      source_files[root].append(os.path.join(root, file))

  base_rst_f = create_sub_rst_file(args, directory)
  base_rst_files = {directory: base_rst_f}

  for base, files in source_files.items():
    if '-' in base:
      print('='*25)
      print(f'Skipping Base: {base}')
      print('='*25)
      continue

    print('='*25)
    print(f'Base: {base}')
    print('='*25)

    if not base in base_rst_files:
      # Check if there are any valid files before creating sub rst
      if not check_directory(base):
        print(f'Skipped {base} because no valid files')
        continue

      # Create intermediate rst
      sub_rst_f = create_sub_rst_file(args, base)
      base_rst_files[base] = sub_rst_f

    # Create sub rst files
    for file in files:
      print(f'File: {file}')

      base, filename = os.path.split(file)
      base_f = base_rst_files[base]
      base_parent = str(Path(base).parent)

      path = file.replace('.py', '')
      module = '.'.join(path.split('/'))
      rubric = path.replace(base_parent + '/', '')

      if file.endswith('.py'):
        base_f.write(f'.. automodule:: {module}\n.. rubric:: :doc:`{rubric}`\n\n.. autosummary::\n    :nosignatures:\n\n')

      # Create rst for source file
      if file.endswith('.py'):
        create_source_rst_file(file, base_f)
      else:
        create_markdown_rst_file(file, base_f)

      base_f.write('\n')

  for fileio in base_rst_files.values():
    fileio.close()

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Script for generating sphinx doc files')
  parser.add_argument('--dirs', default=['ICLcar_env', 'src'], type=str, nargs='+', help='directory to generate doc files for')
  parser.add_argument('--doc-base-dir', default='docs/source', type=str, help='base directory for storing documentation files')

  args, unknown = parser.parse_known_args()
  args = vars(args)

  for directory in args['dirs']:
    generate_doc_for_dir(args, directory)