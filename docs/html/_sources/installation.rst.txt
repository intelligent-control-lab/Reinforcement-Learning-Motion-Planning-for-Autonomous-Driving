Installation
============

Setup instructions
==================

Clone repo:
`git clone --recursive https://github.com/aliang8/icl-safe-driving.git`

`git submodule update --init --recursive`

`cd spinningup/` and `pip3 install -e .`

### Screenshot of car simulation environment

1. Create virtualenv for development
    - `./setup [virtualenv name] [cpu/gpu]`

2. Running the simulator in manual mode:
    - `python3 gym-ICLcar/gym_ICLcar/test_car_gym.py -es exp_settings/default_settings.yaml exp_settings/human_settings.yaml`

3. Install the gym environment:
    - `pip install -e gym-ICLcar`

4. Test the gym env:
    - `python3 gym-ICLcar/gym_ICLcar/test_car_gym.py`

5. Try training a DDPG agent.
    - `python3 -m ipdb -c continue src/train.py -es exp_settings/default_settings.yaml exp_settings/exp_settings.yaml`

6. Train DDPG agent with SpinningUp implementation.
    - `python3 -m ipdb -c continue src/external/spinup_main.py -es exp_settings/default_settings.yaml exp_settings/exp_settings.yaml`

### Testing Rllab and Spinup implementations of DDPG

Installing rllab
```
  git clone https://github.com/rll/rllab.git
  ./scripts/setup_osx.sh
  source activate rllab3

  conda create -n rllab3 python=3.5
  # might need to create your own requirements.txt if you run into installation errors
```

Install spinningup
```
  git clone https://github.com/openai/spinningup.git
  cd spinningup
  pip install -e .
```

### Installing miniconda on server

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./miniconda.sh && bash ./miniconda.sh -b -p ./miniconda3 && rm -r ./miniconda.sh
