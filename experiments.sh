# Training
python3 baselines/run.py --alg=ddpg --env=CarRacing-v0 --num_timesteps=1e6 --num_env=4 --network=mlp --layer_norm=True --save_path=./models --log_path=./logs/carracing-v0 --render=False --render_eval=False --nb_rollout_steps=500

# Evaluation
python3 baselines/run.py  --alg=ddpg --env=CarRacing-v0 --num_timesteps=0 --load_path=./models_3/ckpt.pth --play
