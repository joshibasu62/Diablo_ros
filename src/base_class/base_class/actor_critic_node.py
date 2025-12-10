import rclpy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from base_class.reinforcement_learning_node import ReinforcementLearningNode
from std_msgs.msg import Float64
from collections import deque
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # actor head
        self.mean_head = nn.Linear(hidden_size, action_size)
        # learnable log std (one per action dim)
        self.log_std = nn.Parameter(torch.ones(action_size) * -1.0)

        # critic head
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean_head(x)
        mean = torch.clamp(torch.nan_to_num(mean, nan=0.0, posinf=10.0, neginf=-10.0), -10, 10)
        log_std = torch.clamp(torch.nan_to_num(self.log_std, nan=1e-3, posinf=1.0, neginf=1e-3), 1e-3, 1.0)
        value = self.value_head(x).squeeze(-1)
        value = torch.clamp(torch.nan_to_num(value, nan=0.0, posinf=10.0, neginf=-10.0), -10, 10)
        # clamp mean/nan protections if needed upstream

        

        return mean, log_std, value

    def get_action_and_value(self, state):
        mean, log_std, value = self.forward(state)
        
        std = F.softplus(log_std) + 1e-4
        std = torch.clamp(std, 1e-4, 1.0)
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value


class RolloutBuffer:
    """Simple on-policy rollout buffer for A2C/GAE updates."""
    def __init__(self, rollout_length, state_dim, action_dim):
        self.rollout_length = rollout_length
        self.ptr = 0
        self.states = [None] * rollout_length
        self.actions = [None] * rollout_length
        self.log_probs = [None] * rollout_length
        self.rewards = [0.0] * rollout_length
        self.dones = [False] * rollout_length
        self.values = [0.0] * rollout_length
        

    def add(self, state, action, log_prob, reward, done, value):
        idx = self.ptr
        self.states[idx] = state.detach().cpu().numpy() if isinstance(state, torch.Tensor) else np.array(state)
        self.actions[idx] = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else np.array(action)
        self.log_probs[idx] = log_prob.detach().cpu().numpy() if isinstance(log_prob, torch.Tensor) else float(log_prob)
        self.rewards[idx] = float(reward)
        self.dones[idx] = bool(done)
        self.values[idx] = float(value.detach())

        
        self.ptr += 1

    def is_full(self):
        return self.ptr >= self.rollout_length

    def clear(self):
        self.ptr = 0

    def get(self):
        # return numpy arrays trimmed to ptr
        n = self.ptr
        states = np.stack(self.states[:n])
        actions = np.stack(self.actions[:n])
        log_probs = np.array(self.log_probs[:n])
        rewards = np.array(self.rewards[:n])
        dones = np.array(self.dones[:n], dtype=np.bool_)
        values = np.array(self.values[:n])
        return states, actions, log_probs, rewards, dones, values


def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_adv = 0.0
    for t in reversed(range(n)):
        nonterminal = 1.0 - float(dones[t])
        next_value = values[t + 1] if t + 1 < n else last_value
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        advantages[t] = delta + gamma * lam * nonterminal * last_adv
        last_adv = advantages[t]
    returns = advantages + values
    return advantages, returns


class ActorCriticNode(ReinforcementLearningNode):
    def __init__(self, name="actor_critic_node"):
        super().__init__(name)

        # params (use existing param listener values unless overridden)
        sample_obs = self.get_diablo_observations()
        self.state_size = len(sample_obs)
        self.action_size = len(self.max_effort_command)

        # hyperparams 
        self.rollout_length = 2048 
        self.mini_batch_size = 512
        self.update_epochs = 4
        self.gamma = float(self.discount_factor)
        self.gae_lambda = 0.95
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.lr = 3e-4

        # networks
        self.ac = ActorCritic(self.state_size, self.action_size).to(device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=self.lr)

        # buffer
        self.buffer = RolloutBuffer(self.rollout_length, self.state_size, self.action_size)

        # storage for logging
        self.episode_reward = 0.0
        self.episode_length = 0

        # run timer
        self.create_timer(0.05, self.run)  # 20 Hz

    def create_continuous_command(self, action_tensor):
        max_effort = torch.tensor(self.max_effort_command, device=device)
        scaled_action = torch.clamp(action_tensor, -1.0, 1.0) * max_effort
        return scaled_action.detach().cpu().numpy().tolist()

    def run_one_step(self):
        state_np = self.get_diablo_observations()
        state = torch.FloatTensor(state_np).to(device)

        # get action + value
        action_tensor, log_prob, entropy, value = self.ac.get_action_and_value(state.unsqueeze(0))
        # outputs have batch dim=1
        action_tensor = action_tensor.squeeze(0)
        log_prob = log_prob.squeeze(0)
        entropy = entropy.squeeze(0)
        value = value.squeeze(0)

        scaled_action = self.create_continuous_command(action_tensor)
        self.take_action(scaled_action)

        reward = self.compute_reward_from_state(state_np)
        done = self.is_simulation_stopped()

        # print(f'this is step {self.step} and reward is {reward}')

        # store in buffer
        self.buffer.add(state, action_tensor, log_prob, reward, done, value)

        

        self.update_simulation_status()
        self.episode_reward += reward
        self.episode_length += 1
        self.step += 1

        # If episode ended, log and reset episode counters but keep buffer data (on-policy)
        # if done or self.is_episode_ended():
        #     self.get_logger().info(f"Episode {self.episode} ended. length={self.episode_length} reward={self.episode_reward:.3f}")
        #     self.episode += 1
        #     self.episode_length = 0
        #     self.episode_reward = 0.0
        #     # make sure simulation reset is handled by base class
        #     self.restart_learning_loop()

    def compute_reward_from_state(self, state_np):
        # use the same logic as your old reward but operate on provided state snapshot
        height = state_np[16]
        roll = state_np[17]
        pitch = state_np[18]
        to_be_roll = 0.0
        to_be_pitch = 0.0

        reward = 0.0
        reward_for_each_step = 0.5

        if height < self.height_limit_lower and height > self.height_limit_upper:
            reward -= 1.0
        else:
            reward += 5.0

        roll_dist = abs(to_be_roll - roll)
        reward -= roll_dist
        # if abs(roll) > 0.174533:
        #     reward -= 1.0
        # else:
        #     reward += 0.5

        pitch_dist = abs(to_be_pitch - pitch)
        reward -= pitch_dist
        # if abs(pitch) > 0.174533:
        #     reward -= 2.0
        # else:
        #     reward += 6.0

        reward += reward_for_each_step
        return reward

    def finish_update(self):
        # called when buffer full or training triggered
        states_np, actions_np, log_probs_np, rewards_np, dones_np, values_np = self.buffer.get()
        n = len(rewards_np)
        if n == 0:
            return

        # compute last value for bootstrapping
        last_state = torch.FloatTensor(self.get_diablo_observations()).to(device)
        with torch.no_grad():
            _, _, last_value = self.ac.forward(last_state.unsqueeze(0))
            last_value = float(last_value.squeeze(0).cpu().numpy())

        advantages, returns = compute_gae(rewards_np, values_np, dones_np, last_value, gamma=self.gamma, lam=self.gae_lambda)

        # convert to tensors
        states = torch.FloatTensor(states_np).to(device)
        actions = torch.FloatTensor(actions_np).to(device)
        old_log_probs = torch.FloatTensor(log_probs_np).to(device)
        returns_t = torch.FloatTensor(returns).to(device)
        advantages_t = torch.FloatTensor(advantages).to(device)

        # normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # perform multi-epoch updates with mini-batches (on-policy)
        dataset_size = states.shape[0]
        inds = np.arange(dataset_size)

        value_losses = []
        policy_losses = []
        entropies = []

        for epoch in range(self.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, dataset_size, self.mini_batch_size):
                mb_inds = inds[start : start + self.mini_batch_size]
                mb_states = states[mb_inds]
                mb_actions = actions[mb_inds]
                mb_old_log_probs = old_log_probs[mb_inds]
                mb_returns = returns_t[mb_inds]
                mb_adv = advantages_t[mb_inds]

                # forward pass
                mean, log_std, values_pred = self.ac.forward(mb_states)
                std = F.softplus(log_std) + 1e-4
                std = torch.clamp(std, 1e-4, 1.0)
                dist = Normal(mean, std)

                new_log_prob = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                # policy loss (vanilla actor-critic / A2C style)
                policy_loss = -(new_log_prob * mb_adv).mean()
                # value loss
                value_loss = F.mse_loss(values_pred, mb_returns)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.optimizer.step()

                value_losses.append(value_loss.item())
                policy_losses.append(policy_loss.item())
                entropies.append(entropy.item())

        # logging
        self.get_logger().info(f"Update finished: policy_loss={np.mean(policy_losses):.4f} value_loss={np.mean(value_losses):.4f} entropy={np.mean(entropies):.4f} reward={np.mean(rewards_np)}")

        # clear buffer
        self.buffer.clear()

    def run(self):
        if not self.is_simulation_ready():
            return

        if self.stop_run_when_learning_ended():
            return

        # if buffer full -> compute update first (on-policy)
        if self.buffer.is_full():
            self.finish_update()

        # handle episode termination
        if self.is_episode_ended() or self.is_simulation_stopped():
            # ensure final update if buffer has data
            self.finish_update()
            self.restart_learning_loop()
            self.episode += 1
            return

        # normal step
        self.run_one_step()


def main(args=None):
    rclpy.init(args=args)
    node = ActorCriticNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
