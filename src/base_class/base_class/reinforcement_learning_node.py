from std_msgs.msg import Float64, Float64MultiArray
from collections.abc import Callable
from base_class.base_node_class import DiabloBaseNode
from base_class.diablo_reinforcement_learning_parameters import (
    reinforcement_learning_node_parameters,
)
import torch
import pandas as pd

class ReinforcementLearningNode(DiabloBaseNode):
    def __init__(
        self,
        name,
        max_number_of_episodes=None,
        max_number_of_steps=None,
        max_effort_command=None,
        discount_factor=None,
        reward=None,
        optimizer=None,
        loss_function=None,
        model=None,
    ):
        super().__init__(name)
        self.param_listener = reinforcement_learning_node_parameters.ParamListener(self)
        self.parameters = self.param_listener.get_params()
        self.max_number_of_episodes = (
            self.parameters.max_number_of_episodes if max_number_of_episodes is None else max_number_of_episodes
        )
        self.max_number_of_steps = (
            self.parameters.max_number_of_steps if max_number_of_steps is None else max_number_of_steps
        )
        self.max_effort_command = (
            self.parameters.max_effort_command if max_effort_command is None else max_effort_command
        )
        self.discount_factor = self.parameters.discount_factor if discount_factor is None else discount_factor
        self.reward = self.parameters.reward if reward is None else reward
        self.episode = 0
        self.step = 0
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.model = model

    def is_episode_ended(self) -> bool:
        return self.step >= self.max_number_of_steps

    def create_command(self, action: int) -> Float64MultiArray:
        msg = Float64MultiArray()

        if action == 0:
            # use +max_effort for all 8 joints
            msg.data = self.max_effort_command
        else:
            # use -max_effort for all 8 joints
            msg.data = [-x for x in self.max_effort_command]

        return msg

    def stop_run_when_learning_ended(self):
        if self.episode >= self.max_number_of_episodes:
            self.get_logger().info("Maximum episodes reached. Saving model and shutting down.")

            # Save the model (for your A2C node)
            if hasattr(self, "ac"):
                torch.save(self.ac.state_dict(), "actor_critic_final.pth")
                self.get_logger().info("Saved final ActorCritic model to actor_critic_final.pth")

            # Only do this if you actually track episode rewards somewhere
            if hasattr(self, "rewards_to_save_as_csv"):
                rewards_df = pd.DataFrame(self.rewards_to_save_as_csv, columns=["Total Reward"])
                rewards_df.to_csv("rewards.csv", index=False)
                self.get_logger().info("Saved rewards to rewards.csv")

            import rclpy
            rclpy.shutdown()
            return True  # important

        return False

    def advance_episode_when_finished(self, clean_up_function: Callable[[], None] = None):
        
        should_restart = self.is_episode_ended() or self.is_simulation_stopped()
    
        self.get_logger().info(f"Checking episode end: Step={self.step}, Truncated={self.is_simulation_stopped()}, ShouldRestart={should_restart}")
        
        if should_restart:
            self.get_logger().info(f"Ended episode: {self.episode} with score: {self.step}")
            self.episode += 1
            self.step = 0
            self.restart_learning_loop()
            clean_up_function and clean_up_function()
