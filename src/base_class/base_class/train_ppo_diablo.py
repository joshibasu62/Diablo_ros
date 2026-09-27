# train_ppo_diablo.py

import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from base_class.diablo_ppo_env import DiabloEnv
import os


def make_env():
    max_effort_command = [70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0]
    env = DiabloEnv(max_effort_command=max_effort_command)
    env = Monitor(env)
    return env


def main():
    rclpy.init()
    env = DummyVecEnv([make_env])
    log_dir = os.path.join(os.path.expanduser("~"), "ppo_diablo_logs")
    os.makedirs(log_dir, exist_ok=True)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=512,           # like your rollout_length
        batch_size=128,        # like your mini_batch_size
        n_epochs=4,            # PPO epochs
        gamma=0.99,
        gae_lambda=0.95,
        vf_coef=0.5,
        ent_coef=0.003,
        learning_rate=3e-4,
        clip_range=0.2,
        # init log_std=-2.5 -> torque sigma ~5.7 Nm; SB3 default (0.0)
        # gives sigma=70 Nm which topples the robot within ~70 ms
        policy_kwargs={"log_std_init": -2.5},
        verbose=1,
        tensorboard_log=log_dir,
    )

    model.learn(
        total_timesteps=1_000_000,
        tb_log_name="PPO_Diablo",
        )

    model.save("ppo_diablo")

    env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()