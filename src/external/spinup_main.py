import os
import torch
import gym
import yaml
import pickle
import spinup
import torch.nn as nn
import numpy as np
from gym_ICLcar.envs.wrappers import make_wrapped_env

from src.external.actor_critic import ActorCriticDDPG, ActorCriticSAC
from src.external.replay_buffer import DictReplayBuffer
from src.external.tf_logger import TensorboardLogger
from src.utils import *
from spinningup.spinup.utils.run_utils import ExperimentGrid, setup_logger_kwargs

def main(args):
  if args['hide_display']: os.environ["SDL_VIDEODRIVER"] = "dummy"

  mode = args['mode']
  log(f'Mode: {mode}', color='green')

  # ==================================
  # Logger
  # ==================================
  exp_name = args['exp_name']

  logger_kwargs = setup_logger_kwargs(exp_name, args['seed'], data_dir=args['log_dir'])
  output_dir = logger_kwargs['output_dir']
  dir_exists = os.path.exists(output_dir)
  if dir_exists and mode == 'train':
    log(f'Experiment name {exp_name} already used', 'red')
    raise RuntimeError(f'Experiment name {exp_name} already used')

  # ==================================
  # Environment
  # ==================================
  def env_fn(mode):
    if mode == 'test':
      video_kwargs = dict(
        save_frames=args['save_frames'],
        save_dir=os.path.join(output_dir, 'videos'),
        video_save_frequency=args['video_save_frequency']
      )
    else:
      video_kwargs = dict()

    reward_func_kwargs = dict(
      rotation_penalty_weight=args['rotation_penalty_weight'],
      distance_penalty_weight=args['distance_penalty_weight'],
      angle_penalty_weight=args['angle_penalty_weight']
    )

    env_kwargs = dict(
        env_id=args['environment_id'],
        env_mode=args['env_mode'],
        use_textures=args['use_textures'],
        state_sources=args['state_sources'],
        num_future_info=args['num_future_info'],
        reward_func_kwargs=reward_func_kwargs,
        video_kwargs=video_kwargs
    )
    train_envs = make_wrapped_env(rank=0, **env_kwargs)
    train_envs.setup(args)
    return train_envs

  test_env = env_fn('test')
  obs_space, action_space = test_env.observation_space, test_env.action_space
  test_env.close()

  device = torch.device("cuda" if torch.cuda.is_available() and args['use_gpu'] else "cpu")
  log('Running on: {}'.format(device), color='green')

  # ==================================
  # Replay Buffer
  # ==================================
  replay_buffer_kwargs = dict(
    obs_space=obs_space,
    action_space=action_space,
    size=args['buffer_size'],
    device=device
  )

  replay_buffer = DictReplayBuffer

  # ==================================
  # Actor Critic Models
  # ==================================
  encoder_kwargs = dict(
    input_width=args['input_size'],
    input_height=args['input_size'],
    input_channels=args['channels'],
    output_size=args['image_embedding_dimension'],
    kernel_sizes=[4, 4],
    n_channels=[32, 64],
    strides=[2, 2],
    paddings=np.zeros(2, dtype=np.int64),
    hidden_sizes=None,
    added_fc_input_size=0,
    batch_norm_conv=False,
    batch_norm_fc=False,
    init_w=1e-4,
    hidden_init=nn.init.xavier_uniform_,
    hidden_activation=nn.ReLU(),
    # output_activation=identity,
  )

  obs_dim=sum([obs_space.spaces[k].shape[0] for k in obs_space.spaces.keys() if k != 'image'])
  mlp_kwargs = dict(
    sizes=[obs_dim, args['context_embedding_dimension']],
    activation=nn.ReLU()
  )

  ac_kwargs = dict(
    observation_space=obs_space,
    action_space=action_space,
    hidden_sizes=args['hidden_sizes'],
    activation=torch.nn.ReLU,
    encoder_kwargs=encoder_kwargs,
    mlp_kwargs=mlp_kwargs,
    device=device
  )

  # ==================================
  # Algorithm shared arguments
  # ==================================
  shared_kwargs = dict(
    # checkpoint_file=args['checkpoint_file'],
    mode=mode,
    ac_kwargs=ac_kwargs,
    replay_buffer=replay_buffer,
    replay_buffer_kwargs=replay_buffer_kwargs,
    seed=args['seed'],
    steps_per_epoch=args['steps_per_epoch'],
    epochs=args['epochs'],
    replay_size=args['buffer_size'],
    gamma=args['gamma'],
    polyak=args['tau'],
    batch_size=args['batch_size'],
    start_steps=args['start_steps'],
    update_after=args['update_after'],
    update_every=args['update_every'],
    num_test_episodes=args['num_test_episodes'],
    max_ep_len=args['max_episode_length'],
    logger=TensorboardLogger,
    logger_kwargs=logger_kwargs,
    save_freq=1,
    device=device
  )

  # ==================================
  # Single experiment
  # ==================================
  if not args['experiment_grid']:
    if args['algo'] == 'ddpg':
      if args['mode'] == 'train':
        spinup.ddpg_pytorch(
          env_fn,
          actor_critic=ActorCriticDDPG,
          pi_lr=args['actor_lr'],
          q_lr=args['critic_lr'],
          act_noise=args['noise_stddev'],
          **shared_kwargs
        )
    elif args['algo'] == 'sac':
      # import ipdb
      # ipdb.set_trace()
      if args['mode'] == 'train':
        # Save a copy of the argument configs
        os.makedirs(output_dir, exist_ok=True)
        pickle.dump(args, open(os.path.join(output_dir, 'experiment_config.pkl'), 'wb'))

        spinup.sac_pytorch(
          env_fn,
          actor_critic=ActorCriticSAC,
          lr=args['learning_rate'],
          alpha=args['alpha'],
          **shared_kwargs
        )
      else:
        spinup.sac_pytorch_test(
          env_fn,
          actor_critic=ActorCriticSAC,
          **shared_kwargs
        )
    else:
      raise NotImplementedError
  else:
    # ==================================
    # Experiment Grid
    # ==================================
    exp_grid = yaml.load(open(args['experiment_grid'], 'r'))
    grid_keys = exp_grid['grid'].keys()

    eg = ExperimentGrid(name=args['exp_name'])
    del shared_kwargs['logger_kwargs']
    for k, v in shared_kwargs.items():
      if k not in grid_keys:
        eg.add(k, v)

    for k, v in exp_grid['grid'].items():
      eg.add(k, v)

    eg.add('env_fn', env_fn)
    eg.add('actor_critic', actor_critic)

    if args['algo'] == 'ddpg':
      # Add ddpg only arguments
      if 'pi_lr' not in grid_keys:
        eg.add('pi_lr', args['actor_lr'])
      if 'q_lr' not in grid_keys:
        eg.add('q_lr', args['critic_lr'])
      if 'act_noise' not in grid_keys:
        eg.add('act_noise', args['noise_stddev'])
      eg.run(spinup.ddpg_pytorch, num_cpu=args['num_cpu'], data_dir=args['log_dir'])
    elif args['algo'] == 'sac':
      pass

if __name__ == "__main__":
  from src.configs import add_experiment_args, add_logging_args, add_training_args, add_rl_agent_args, add_encoder_args, add_car_env_args, add_spinning_up_args
  from pprint import pprint
  import argparse
  from src.utils import load_experiment_settings
  parser = argparse.ArgumentParser(description='DDPG for CarEnv')
  parser = add_experiment_args(parser)
  parser = add_logging_args(parser)
  parser = add_training_args(parser)
  parser = add_rl_agent_args(parser)
  parser = add_encoder_args(parser)
  parser = add_car_env_args(parser)
  parser = add_spinning_up_args(parser)

  args, unknown = parser.parse_known_args()
  args = vars(args)

  experiment_settings = load_experiment_settings(args['experiment_settings'])
  args.update(experiment_settings)

  pprint(args)
  main(args)

