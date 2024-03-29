import sys
import gym
import os
import random
import numpy as np
import gym_ICLcar
from envs.env_v2.env_configs import *
from envs.env_v2.wrappers import make_wrapped_env
import ipdb

if __name__ == '__main__':
  from src.configs import add_experiment_args, add_training_args, add_rl_agent_args, add_car_env_args, add_spinning_up_args, add_logging_args
  from pprint import pprint
  import argparse
  from src.utils import load_experiment_settings

  # ==============================
  # Loading environment settings
  # ==============================
  parser = argparse.ArgumentParser(description='CarEnv Manual Mode')
  parser = add_experiment_args(parser)
  parser = add_logging_args(parser)
  parser = add_car_env_args(parser)
  parser = add_rl_agent_args(parser)
  args, unknown = parser.parse_known_args()
  args = vars(args)

  experiment_settings = load_experiment_settings(args['experiment_settings'])
  args.update(experiment_settings)

  pprint(args)

  random.seed(args['seed'])
  np.random.seed(args['seed'])

  if args['hide_display']: os.environ["SDL_VIDEODRIVER"] = "dummy"

  # ==================================
  # Environment
  # ==================================
  # import ipdb; ipdb.set_trace()
  env_kwargs = dict(
      env_id=args['environment_id'],
      env_mode=args['env_mode'],
      use_textures=args['use_textures'],
      state_sources=args['state_sources'],
      num_future_info=args['num_future_info'],
      reward_func_kwargs = dict(
        velocity_reward_weight=args['velocity_reward_weight'],
        rotation_penalty_weight=args['rotation_penalty_weight'],
        distance_penalty_weight=args['distance_penalty_weight'],
        angle_penalty_weight=args['angle_penalty_weight'],
        stationary_penalty_weight=args['stationary_penalty_weight']
      ),
      video_kwargs=dict(
        save_frames=args['save_frames'],
        save_dir=args['log_dir'],
        video_save_frequency=args['video_save_frequency']
      )
  )
  env = make_wrapped_env(rank=0, **env_kwargs)
  env.setup(args)
  env.reset()

  while True:
      state, reward, done, info = env.step(None, mode=args['env_mode'])
      # import ipdb; ipdb.set_trace()
      if done: env.reset()
      env.render()
      # import ipdb; ipdb.set_trace()

  print('Successfully run gym environment.')