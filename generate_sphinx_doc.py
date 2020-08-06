import os
import ast
import fnmatch
import shutil
from pathlib import Path
import collections

files_to_ignore = ['rllab_ddpg.py', '__init__.py', '*.pyc', '*.md', '*.sh', '*.ipynb', '*.json', '*.yaml', '*.pkl', '*.png']

dirs_to_ignore = ['__pycache__']

def check_directory(dir_):
  for _, _, files in os.walk(dir_):
    for bad in files_to_ignore:
      filtered = [f for f in files if not fnmatch.fnmatch(f, bad)]
    if len(filtered) == 0:
      return False
  return True

def create_sub_rst_file(args, base):
  sub_rst_file = os.path.join(args['doc_base_dir'], base + '.rst')

  parent, fn = os.path.split(sub_rst_file)
  if not os.path.exists(parent):
    os.makedirs(parent)

  sub_rst_f = open(sub_rst_file, 'w')
  sub_rst_f.write(f':mod:`{base}`\n')
  sub_rst_f.write('='*50)
  sub_rst_f.write('\n\n')

  module = '.'.join(base.split('/'))
  sub_rst_f.write(f'.. automodule:: {module}\n')
  sub_rst_f.write('.. toctree::\n\n')

  base_parent = str(Path(base).parent)
  for root, dirs, files_ in os.walk(base):
    for bad in files_to_ignore:
      files_ = [f for f in files_ if not fnmatch.fnmatch(f, bad)]
    for bad in dirs_to_ignore:
      dirs = [d for d in dirs if not fnmatch.fnmatch(d, bad)]

    for file in files_:
      file = os.path.join(root, file)
      path = file.replace('.py', '').replace(args['remove_string'], '')
      path = path.replace(base_parent + '/', '')
      sub_rst_f.write(f'    {path}\n')

    for d in dirs:
      path = os.path.join(root, d)
      if not check_directory(path): continue
      path = path.replace(base_parent + '/', '')
      sub_rst_f.write(f'    {path}\n')
    break

  sub_rst_f.write('\n')
  return sub_rst_f

def generate_doc_for_dir(args, dir_):
  # Get paths to generate documentation for
  # and group them by base directory
  source_files = collections.defaultdict(list)
  for root, dirs, files in os.walk(dir_):
    for bad in files_to_ignore:
      files = [f for f in files if not fnmatch.fnmatch(f, bad)]

    for file in files:
      source_files[root].append(os.path.join(root, file))

  # Create base rst
  # doc_dir = args['dir'].replace(args['remove_string'], '')
  # base = os.path.join(args['doc_base_dir'], doc_dir)
  # if os.path.exists(base):
  #   shutil.rmtree(base)
  # os.makedirs(base)
  base_rst_f = create_sub_rst_file(args, dir_)
  base_rst_files = {}
  base_rst_files[dir_] = base_rst_f

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

      with open(file) as f:
        node = ast.parse(f.read())

      functions = [n for n in node.body if isinstance(n, ast.FunctionDef)]
      classes = [n for n in node.body if isinstance(n, ast.ClassDef)]

      base, filename = os.path.split(file)
      (header, ext) = os.path.splitext(filename)
      path = file.replace('.py', '').replace(args['remove_string'], '')
      module = '.'.join(path.split('/'))

      try:
        exec(f'import {module}')
      except:
        print(f'Could not import: {file}')
        continue

      base_f = base_rst_files[base]
      base_f.write(f'.. automodule:: {module}\n')
      base_parent = str(Path(base).parent)
      rubric = path.replace(base_parent + '/', '')
      base_f.write(f'.. rubric:: :doc:`{rubric}`\n\n')
      base_f.write(f'.. autosummary::\n')
      base_f.write(f'    :nosignatures:\n\n')


      # Write sub rst file
      rst_file = os.path.join(args['doc_base_dir'], file.replace(ext, "").replace(args['remove_string'], '') + '.rst')
      root, file = os.path.split(rst_file)
      if not os.path.exists(root):
        os.makedirs(root)

      print(f'Writing {rst_file}')

      with open(rst_file, 'w') as rst:
        # Write header
        rst.write(f'{header} \n')
        rst.write('-'*50)
        rst.write('\n\n')

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

          # methods = [n for n in class_.body if isinstance(n, ast.FunctionDef)]
          # for method in methods:
              # show_info(method)

      base_f.write('\n')

  for fileio in base_rst_files.values():
    fileio.close()

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Script for generating sphinx doc files')
  parser.add_argument('--dirs', default=['src', 'thor', 'testing', 'evaluation'], type=str, nargs='+', help='directory to generate doc files for')
  parser.add_argument('--remove-string', default='', type=str, help='remove this from string when generating docs')
  parser.add_argument('--doc-base-dir', default='docs/source', type=str, help='base directory for storing documentation files')

  args, unknown = parser.parse_known_args()
  args = vars(args)

  for dir_ in args['dirs']:
    generate_doc_for_dir(args, dir_)
