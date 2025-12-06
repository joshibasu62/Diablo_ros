import numpy as np

from collections import deque

import matplotlib.pyplot as plt
#matplotlib inline

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


from base_class.base_node_class import DiabloBaseNode
from base_class.reinforcement_learning_node import ReinforcementLearningNode
import rclpy

class Reinforce(ReinforcementLearningNode):
    def __init__(self):
        super().__init__("reinforce")
        self.create_timer(0.05, self.run)
        # self.policy = Policy(s_size, a_size, 64).to(device)
        # self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        self.episode = 0
        self.step = 0

    def run_one_step(self):
        state = self.get_diablo_observations()
        print(f"state: {state}")
        # action, log_prob = self.policy.act(state)
        # self.take_action(self.create_command(action))
        self.step += 1

    def run(self):
        if not self.is_simulation_ready():
            return

        self.stop_run_when_learning_ended()
        self.advance_episode_when_finished()
        self.run_one_step()



def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(Reinforce())
    rclpy.shutdown()


if __name__ == "__main__":
    main()