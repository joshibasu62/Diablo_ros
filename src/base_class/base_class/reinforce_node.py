import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from std_msgs.msg import Float64MultiArray
from base_class.reinforcement_learning_node import ReinforcementLearningNode
import rclpy
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Policy Network for Continuous Actions

class ContinuousPolicy(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean_head = nn.Linear(hidden_size, action_size)
        self.log_std = nn.Parameter(torch.zeros(action_size)) 

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean_head(x)
        return mean, self.log_std

    def act(self, state):
        # Forward pass
        mean, log_std = self.forward(state)
        
        mean = torch.clamp(torch.nan_to_num(mean, nan=0.0, posinf=10.0, neginf=-10.0), -10, 10)
        std = F.softplus(log_std) + 1e-3  
        std = torch.clamp(torch.nan_to_num(std, nan=1e-3, posinf=1.0, neginf=1e-3), 1e-3, 1.0)
        
        # Create Normal distribution
        dist = Normal(mean, std)
        
        action = dist.rsample()  
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob




class ReinforceContinuousNode(ReinforcementLearningNode):
    def __init__(self, name="reinforce_continuous_node"):
        super().__init__(name)

        # Initialize policy and optimizer
        sample_obs = self.get_diablo_observations()
        self.state_size = len(sample_obs)
        self.action_size = len(self.max_effort_command)  # 8 joints
        self.policy = ContinuousPolicy(self.state_size, self.action_size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)

        # Storage for REINFORCE
        self.states = []
        self.log_probs = []
        self.rewards = []

        self.rewards_to_save_as_csv = []
        self.loss_to_save_as_csv = []

        # Timer
        self.create_timer(0.05, self.run)  # 20 Hz

    
    
    def create_continuous_command(self, action_tensor):
        # Scale each joint action by max_effort_command
        max_effort = torch.tensor(self.max_effort_command, device=device)
        scaled_action = torch.clamp(action_tensor, -1.0, 1.0) * max_effort
        
        return scaled_action.detach().cpu().numpy().tolist()

    
    # Step: get state, act, publish
    
    def run_one_step(self):
        state_np = self.get_diablo_observations()
        state = torch.FloatTensor(state_np).to(device)
        action_tensor, log_prob = self.policy.act(state)
        
        #for debugging
        # print(action_tensor)

        
        scaled_action = self.create_continuous_command(action_tensor)
        self.take_action(scaled_action)

        # Store for REINFORCE update
        self.states.append(state)
        self.log_probs.append(log_prob)
        reward = self.compute_reward()
        self.rewards.append(reward)

        self.update_simulation_status()

        self.step += 1
        # self.get_logger().info(f"Step {self.step}, Reward: {reward:.3f}")

    
    # Reward function
    
    def compute_reward(self):
        reward = 0.0

        reward_for_each_step = 0.5
        # reward for staying above height limit and small roll/pitch
        # reward = 0.0
        height = self.get_diablo_observations()[16]
        roll = self.get_diablo_observations()[17]
        pitch = self.get_diablo_observations()[18]
        vertical_acceleration = self.get_diablo_observations()[19]

        if height < self.height_limit_lower or height > self.height_limit_upper:
            reward -= 1.0
        else:
            reward += 5.0  # small bonus for staying within height limits

        if abs(roll) > 0.174533:  # 10 degrees in radians
            reward -= 1.0
        else:
            reward += 0.5  # small bonus for small roll

        if abs(pitch) > 0.174533: #10 degrees in radians  
            reward -= 2.0  
        else:
            reward += 6.0  # small bonus for small pitch

        if vertical_acceleration < 9.81:
            reward -= 2   # penalize for downward acceleration
        else:
            reward += 6  # small bonus for upward or stable acceleration

        reward += reward_for_each_step
        return reward
        # if height < self.height_limit or roll > 0.174533 or pitch > 0.349066:
        #     return -1.0
        # else:
        #     return 0.1

    
    # Update policy at episode end
    def finish_episode(self):
        if len(self.rewards) == 0:
            self.get_logger().warn("No steps taken this episode. Skipping policy update.")
            self.step = 0
            # self.episode += 1
            return
        G = 0
        returns = []

        # Compute discounted rewards
        for r in reversed(self.rewards):
            G = r + self.discount_factor * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        if returns.std() < 1e-6:
            returns = returns - returns.mean()  # avoid division by tiny std
        else:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)


        # REINFORCE loss
        policy_loss = [-log_prob * R for log_prob, R in zip(self.log_probs, returns)]
        loss = torch.stack(policy_loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #total steps in episode
        self.get_logger().info(f"Total steps in episode: {len(self.rewards)}")

        #total reward in episode
        self.get_logger().info(f"Total reward in episode: {sum(self.rewards):.3f}")

        self.get_logger().info(f"Episode {self.episode} finished. Loss: {loss.item():.3f}")

        # Clear storage
        self.states, self.log_probs, self.rewards = [], [], []
        
        self.step = 0
        
        # self.episode += 1
        # Save rewards and loss for CSV
        self.rewards_to_save_as_csv.append(sum(self.rewards))
        self.loss_to_save_as_csv.append(loss.item())

    
    # Main loop
    
    def run(self):
        if not self.is_simulation_ready():
            return
        
        if self.stop_run_when_learning_ended():
            return

        if self.is_episode_ended() or self.is_simulation_stopped():
            self.finish_episode()
            self.restart_learning_loop()
            self.episode += 1
            return

        self.run_one_step()

def main(args=None):
    import rclpy
    rclpy.init(args=args)
    node = ReinforceContinuousNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
