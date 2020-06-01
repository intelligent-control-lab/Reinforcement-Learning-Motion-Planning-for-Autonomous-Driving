#!/bin/bash

# sample usage: ./setup.sh [name_of_env] [cpu/gpu].yaml

if [[ $1 = "" ]]; then
  echo "Please set the first argument to the name of the conda environment you'd like to set this as"
  exit
fi
name=$1

if [[ $2 = "cpu" ]]; then
  echo "Installing for CPU"
  yaml_file=virtualenv/cpu.yaml
elif [[ $2 = "gpu" ]]; then
  echo "Installing for GPU"
  yaml_file=virtualenv/gpu.yaml
else
  echo "Please set the second argument to either 'cpu' or 'gpu'"
  exit
fi

conda create --name ${name} --yes
source activate ${name}

conda env update --file ${yaml_file}

# Install Pygame learning env

git clone https://github.com/ntasfi/PyGame-Learning-Environment.git
cd PyGame-Learning-Environment/
pip install -e .

echo
echo
echo "This has created an environment '${name}' which you can use for development"
echo "Remove environment with 'conda remove --name ${name} --all'"
