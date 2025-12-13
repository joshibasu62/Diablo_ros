# train_ppo_diablo.py

import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from base_class.diablo_ppo_env import DiabloEnv
import os


def make_env():
    # max_effort_command should match your ROS parameters
    # Example: 8 joints with symmetric torque limits
    max_effort_command = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
    return DiabloEnv(max_effort_command=max_effort_command)


def main():
    # Start ROS2
    rclpy.init()

    # Wrap env in VecEnv for Stable-Baselines3
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
        verbose=1,
        tensorboard_log=log_dir,
    )

    # Train for some timesteps (adjust as needed)
    model.learn(
        total_timesteps=1_000_000,
        tb_log_name="PPO_Diablo",
        )

    model.save("ppo_diablo")

    env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()