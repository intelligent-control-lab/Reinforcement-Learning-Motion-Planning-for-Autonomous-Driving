from gym.envs.registration import register
import gym

found = False
for env in gym.envs.registry.env_specs:
  if 'ICLcar-v0' in env:
    found = True

if not found:
  register(
      id='ICLcar-v0',
      entry_point='gym_ICLcar.envs:ICLcarEnv',
  )