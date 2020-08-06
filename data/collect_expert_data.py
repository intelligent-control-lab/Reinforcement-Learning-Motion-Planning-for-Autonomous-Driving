import os
import pickle
import gym_ICLcar
import pygame as pg
from gym_ICLcar.envs.wrappers import make_wrapped_env


# ==================================
# Environment
# ==================================
def build_env(args):
  reward_func_kwargs = dict(
    rotation_penalty_weight=args['rotation_penalty_weight'],
    distance_penalty_weight=args['distance_penalty_weight']
  )

  env_kwargs = dict(
      env_id=args['environment_id'],
      env_mode='human',
      use_textures=args['use_textures'],
      state_sources=args['state_sources'],
      num_future_info=args['num_future_info'],
      reward_func_kwargs=reward_func_kwargs,
      video_kwargs={}
  )
  train_envs = make_wrapped_env(rank=0, **env_kwargs)
  train_envs.setup(args)
  train_envs.reset()
  return train_envs

def handle_key_events():
  for event in pg.event.get():
    if event.type == pg.QUIT:
      pg.quit()
      return True
    if event.type == pg.KEYDOWN:
      if event.key == pg.K_q:
        pg.quit()
        return True
  return False

def collect_expert_data(args, env):
  data = []
  while True:
    done = handle_key_events()
    if done: break

    state, reward, done, info = env.step(None, mode='human')
    env.render()
    data.append((state, reward, done, info))

  savepath = args['data_file']
  print(f'Done collecting data. Saving to {savepath}. Length of trajectory: {len(data)}')
  pickle.dump(data, open(savepath, 'wb'))

def load_and_replay_data(args, env):
  data_file = args['data_file']

  if not os.path.exists(data_file):
    raise RuntimeError(f"{data_file} does not exist! Please generate the data first.")

  data = pickle.load(open(data_file, 'rb'))

  for t, tup in enumerate(data):
    state, reward, done, info = tup
    action = info['action']

    if args['verbose']:
      print(f'Timestep: {t}, Action: {action}, Done: {done}, Reward: {reward}')

    env.step(action, mode='data_collection')
    env.render()

  print(f'Done replaying data from {savepath}')


if __name__ == "__main__":
  from src.configs import add_experiment_args, add_training_args, add_rl_agent_args, add_car_env_args, add_spinning_up_args
  from pprint import pprint
  import argparse
  from src.utils import load_experiment_settings
  parser = argparse.ArgumentParser(description='Collecting expert data for Imitation Learning')
  parser = add_experiment_args(parser)
  parser = add_training_args(parser)
  parser = add_rl_agent_args(parser)
  parser = add_car_env_args(parser)
  parser = add_spinning_up_args(parser)
  parser.add_argument('--type', default=0, type=int, help='0 - generating data, 1 - replaying data')
  parser.add_argument('--data-file', default="data/expert_trajectory.pkl", type=str, help='where to store the expert data')

  args, unknown = parser.parse_known_args()
  args = vars(args)

  experiment_settings = load_experiment_settings(args['experiment_settings'])
  args.update(experiment_settings)

  pprint(args)

  env = build_env(args)

  if args['type'] == 0:
    collect_expert_data(args, env)
  elif args['type'] == 1:
    load_and_replay_data(args, env)

