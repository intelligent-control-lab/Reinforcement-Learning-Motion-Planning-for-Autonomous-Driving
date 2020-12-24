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

echo "Creating conda environment"
conda create --name ${name} --yes
source activate ${name}

conda env update --file ${yaml_file}

# Install donkey-env

# # git clone https://github.com/openai/baselines
# cd baselines
# git checkout donkey_env
# # git checkout 5b41c926c7a852df3f0928afdf2429f96a3965cb -b compatible
# pip install -e .
# cd ../

echo "Installing PyGame-Learning-Environment"
git clone https://github.com/ntasfi/PyGame-Learning-Environment.git
cd PyGame-Learning-Environment/
pip3 install -e .


echo "Installing requirements.txt"
cd ..
pip3 install -r requirements.txt

echo "Installing gym environment"
pip3 install -e ICLcar_env

echo
echo
echo "This has created an environment '${name}' which you can use for development"
echo "Remove environment with 'conda remove --name ${name} --all'"