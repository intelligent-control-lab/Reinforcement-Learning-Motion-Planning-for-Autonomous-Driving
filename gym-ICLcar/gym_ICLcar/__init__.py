from gym.envs.registration import register

register(
    id='ICLcar-v0',
    entry_point='gym_ICLcar.envs:ICLcarEnv',
)