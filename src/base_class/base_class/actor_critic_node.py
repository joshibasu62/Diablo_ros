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
        
        log_std = torch.nan_to_num(self.log_std, nan=-0.7, posinf=1.0, neginf=-5.0)
        log_std = torch.clamp(log_std, -5.0, 1.0)
        # log_std = torch.clamp(torch.nan_to_num(self.log_std, nan=1e-3, posinf=1.0, neginf=1e-3), 1e-3, 1.0)
        
        value = self.value_head(x).squeeze(-1)
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        # value = torch.clamp(torch.nan_to_num(value, nan=0.0, posinf=10.0, neginf=-10.0), -10, 10)
        # clamp mean/nan protections if needed upstream

        

        return mean, log_std, value

    def get_action_and_value(self, state):
        mean, log_std, value = self.forward(state)
        
        # std = F.softplus(log_std) + 1e-4
        # std = torch.clamp(std, 1e-4, 1.0)

        std = torch.exp(log_std).clamp(1e-3, 1.0)

        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value


class RolloutBuffer:
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

    def has_data(self):
        return self.ptr > 0

    def clear(self):
        self.ptr = 0

    def get(self):
        n = self.ptr
        # assume n > 0; caller must check has_data()
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
        self.rollout_length = 1048 
        self.mini_batch_size = 1048
        self.update_epochs = 1
        self.gamma = float(self.discount_factor)
        self.gae_lambda = 0.95
        self.value_coef = 0.5
        self.entropy_coef = 0.003
        self.lr = 3e-4


        self.reward_weights = {
            'orient': 2.0,
            'height': 1.5,
            'com': 1.0,
            'clearance': 0.5,
            'wheel_slip': -0.05,
            'wheel_torque': -0.01,
            'joint_torque': -0.001,
            'action_smooth': -0.01,
            'energy': -0.0001
        }
        
        # Standing parameters
        self.target_height = (self.height_limit_lower + self.height_limit_upper) / 2.0
        self.standing_height = 0.6  # Adjust based on your robot
        self.max_roll_pitch = np.deg2rad(15)
        
        # Wheel indices 
        
        self.wheel_indices = [6, 7]  
        
       
        self.prev_actions = np.zeros(self.action_size)
        self.prev_joint_velocities = np.zeros(8)
        
        
        self.upright_time_steps = 0
        self.required_upright_steps = 50
        
        
        self.episode = 0
        self.curriculum_stage = 1 




        # networks
        self.ac = ActorCritic(self.state_size, self.action_size).to(device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=self.lr)

        # buffer
        self.buffer = RolloutBuffer(self.rollout_length, self.state_size, self.action_size)

        # storage for logging
        self.episode_reward = 0.0
        self.episode_length = 0
        self.step = 0

        # run timer
        self.create_timer(0.005, self.run)  # 20 Hz

    def create_continuous_command(self, action_tensor):
        max_effort = torch.tensor(self.max_effort_command, device=device)
        scaled_action = torch.clamp(action_tensor, -1.0, 1.0) * max_effort
        return scaled_action.detach().cpu().numpy().tolist()
    

    def estimate_foot_height(self, leg_joint_positions, is_left=True):
        """
        Simplified foot height estimation from joint positions
        Adjust based on your robot's kinematics
        """
        hip_angle = leg_joint_positions[0]
        knee_angle = leg_joint_positions[1]
        ankle_angle = leg_joint_positions[2]
        
        # Very simplified model - adjust lengths based on your robot
        thigh_length = 0.2
        shin_length = 0.2
        
        # Calculate foot position relative to hip
        foot_height = thigh_length * np.sin(hip_angle) + shin_length * np.sin(hip_angle + knee_angle)
        
        # Ankle adjustment (simplified)
        foot_height += 0.05 * np.sin(hip_angle + knee_angle + ankle_angle)
        
        return abs(foot_height)

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

        reward = self.compute_reward_from_state(state_np, actions_np=action_tensor.detach().cpu().numpy())
        done = self.is_simulation_stopped()

        # print(f'this is step {self.step} and reward is {reward}')

        # store in buffer
        self.buffer.add(state, action_tensor, log_prob, reward, done, value)

        

        self.update_simulation_status()
        self.episode_reward += reward
        self.episode_length += 1
        self.step += 1

    

    # def compute_reward_from_state(self, state_np):
    #     height = state_np[16]
    #     roll = state_np[17]
    #     pitch = state_np[18]

    #     target_height = (self.height_limit_lower + self.height_limit_upper) / 2.0
    #     alive_bonus = 0.1

    #     reward = 0.0

    #     # 1) height reward
    #     if height < self.height_limit_lower or height > self.height_limit_upper:
    #         reward -= 5.0   # clearly bad height
    #     else:
    #         # smoothly reward being near target height
    #         height_error = height - target_height
    #         reward += 3.0 * np.exp(- (height_error / 0.05) ** 2)  # Gaussian around target

    #     # 2) orientation penalty (rad^2 gives stronger penalty when angle grows)

        
    #     angle_penalty_scale = 3.0
    #     reward -= angle_penalty_scale * (roll ** 2 + pitch ** 2)

    #     # 3) small alive bonus each step
    #     reward += alive_bonus

    #     # 4) optional: strong penalty on failure
    #     if self.is_simulation_stopped():
    #         reward -= 20.0

    #     return reward

    def compute_reward_from_state(self, state_np, actions_np=None, dt=0.005):
        """
        Comprehensive reward function for wheel-ankle biped standing up
        """
        # Extract observations
        joint_positions = state_np[0:8]
        joint_velocities = state_np[8:16]
        ground_distance = state_np[16]  # Lidar distance
        roll = state_np[17]
        pitch = state_np[18]
        vertical_accel = state_np[19]  # az
        
    
        if actions_np is None:
            actions_np = np.zeros(self.action_size)
        
        
        k1, k2 = 2.0, 2.0
        R_orient = np.exp(-k1 * abs(roll) - k2 * abs(pitch))
        
        if abs(roll) < 0.1 and abs(pitch) < 0.1: 
            R_orient += 0.5
        
        # 2. Height Reward (R_height)
        
        if ground_distance < self.height_limit_lower:
            height_progress = 0.0
        else:
            height_progress = (ground_distance - self.height_limit_lower) / \
                             (self.standing_height - self.height_limit_lower)
        height_progress = np.clip(height_progress, 0, 1)
        
        # Gaussian reward around target height
        height_error = ground_distance - self.target_height
        R_height_gaussian = np.exp(-(height_error / 0.05) ** 2)
        
        # Combined height reward
        R_height = 0.7 * height_progress + 0.3 * R_height_gaussian
        
        
        if self.height_limit_lower < ground_distance < self.height_limit_upper:
            R_com = 1.0
        else:
            R_com = -1.0
        
        # 4. Foot Clearance Reward 
        if ground_distance > self.height_limit_lower + 0.1:
            left_foot_height = self.estimate_foot_height(joint_positions[0:4], True)
            right_foot_height = self.estimate_foot_height(joint_positions[4:8], False)
            target_foot_height = 0.02  # 2cm off ground
            
            clearance_reward_left = np.exp(-10 * abs(left_foot_height - target_foot_height))
            clearance_reward_right = np.exp(-10 * abs(right_foot_height - target_foot_height))
            R_clearance = 0.5 * (clearance_reward_left + clearance_reward_right)
        else:
            R_clearance = 0.0
        
        # 5. Wheel Slip Penalty 
        wheel_velocities = joint_velocities[6:8]  # Assuming last 2 are wheels
        
        if ground_distance < self.target_height:
           
            P_wheel_slip = -np.sum(np.abs(wheel_velocities))
        else:
            
            P_wheel_slip = -0.1 * np.sum(np.abs(wheel_velocities))
        
        # 6. Wheel Torque Penalty 
        wheel_torques = actions_np[6:8]
        P_wheel_torque = -np.sum(np.abs(wheel_torques))
        
        # 7. Joint Torque Penalty 
        joint_torques = actions_np[0:8]
        P_joint_torque = -np.sum(np.square(joint_torques))
        
        # 8. Action Smoothness Penalty
        action_changes = actions_np - self.prev_actions
        P_action_smooth = -np.sum(np.square(action_changes))
        
        # 9. Joint Acceleration Penalty
        joint_velocities_np = np.asarray(joint_velocities, dtype=np.float32)
        prev_joint_velocities_np = np.asarray(self.prev_joint_velocities, dtype=np.float32)

        joint_accelerations = (joint_velocities_np - prev_joint_velocities_np) / dt

        P_joint_accel = -0.001 * np.sum(np.square(joint_accelerations))
        
        # 10. Energy Efficiency Penalty
        power = np.sum(np.abs(joint_torques * joint_velocities))
        P_energy = -0.0001 * power
        
        # 11. Success Bonus (sparse reward)
        success_bonus = 0.0
        is_upright = (abs(roll) < 0.17 and abs(pitch) < 0.17 and
                     self.height_limit_lower < ground_distance < self.height_limit_upper)
        
        if is_upright:
            self.upright_time_steps += 1
            if self.upright_time_steps >= self.required_upright_steps:
                success_bonus = 100.0
                self.get_logger().info(f"Episode {self.episode}: SUCCESS! Standing achieved!")
        else:
            self.upright_time_steps = 0
        
        # 12. Failure Penalty
        failure_penalty = 0.0
        if self.is_simulation_stopped():
            failure_penalty = -20.0
        elif abs(roll) > np.deg2rad(45) or abs(pitch) > np.deg2rad(45):
            failure_penalty = -10.0
        
        # Update previous values
        self.prev_actions = actions_np.copy()
        self.prev_joint_velocities = joint_velocities.copy()
        
        # Calculate total reward with curriculum
        if self.curriculum_stage == 1:
            # Stage 1: Basic height and orientation
            total_reward = (
                self.reward_weights['orient'] * R_orient +
                self.reward_weights['height'] * R_height +
                success_bonus +
                failure_penalty
            )
        elif self.curriculum_stage == 2:
            # Stage 2: Add wheel control
            total_reward = (
                self.reward_weights['orient'] * R_orient +
                self.reward_weights['height'] * R_height +
                self.reward_weights['wheel_slip'] * P_wheel_slip +
                self.reward_weights['wheel_torque'] * P_wheel_torque +
                success_bonus +
                failure_penalty
            )
        else:
            # Stage 3: Full reward system
            total_reward = (
                self.reward_weights['orient'] * R_orient +
                self.reward_weights['height'] * R_height +
                self.reward_weights['com'] * R_com +
                self.reward_weights['clearance'] * R_clearance +
                self.reward_weights['wheel_slip'] * P_wheel_slip +
                self.reward_weights['wheel_torque'] * P_wheel_torque +
                self.reward_weights['joint_torque'] * P_joint_torque +
                self.reward_weights['action_smooth'] * P_action_smooth +
                self.reward_weights['energy'] * P_energy +
                success_bonus +
                failure_penalty
            )
        
        # Progress curriculum
        if self.episode > 0 and self.episode % 100 == 0:
            if self.curriculum_stage < 3:
                self.curriculum_stage += 1
                self.get_logger().info(f"Advancing to curriculum stage {self.curriculum_stage}")
        
        # Debug logging
        # if np.random.random() < 0.01:  # Log 1% of steps
        #     self.get_logger().info(
        #         f"Ep {self.episode}, Step {self.step}: "
        #         f"Height={ground_distance:.3f}, "
        #         f"Roll={roll:.3f}, Pitch={pitch:.3f}, "
        #         f"Reward={total_reward:.3f}"
        #     )
        
        return total_reward

    def finish_update(self):
        # Safety: do nothing if buffer is empty
        if not self.buffer.has_data():
            self.get_logger().warn("finish_update() called but rollout buffer is empty; skipping.")
            return

        states_np, actions_np, log_probs_np, rewards_np, dones_np, values_np = self.buffer.get()

        # compute last value for bootstrapping
        last_state = torch.FloatTensor(self.get_diablo_observations()).to(device)
        with torch.no_grad():
            _, _, last_value = self.ac.forward(last_state.unsqueeze(0))
            last_value = float(last_value.squeeze(0).cpu().numpy())

        advantages, returns = compute_gae(
            rewards_np, values_np, dones_np, last_value,
            gamma=self.gamma, lam=self.gae_lambda
        )

        # convert to tensors
        states = torch.FloatTensor(states_np).to(device)
        actions = torch.FloatTensor(actions_np).to(device)
        old_log_probs = torch.FloatTensor(log_probs_np).to(device)
        returns_t = torch.FloatTensor(returns).to(device)
        advantages_t = torch.FloatTensor(advantages).to(device)

        # normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        dataset_size = states.shape[0]
        inds = np.arange(dataset_size)

        value_losses, policy_losses, entropies, total_loss = [], [], [], []

        for epoch in range(self.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, dataset_size, self.mini_batch_size):
                mb_inds = inds[start:start + self.mini_batch_size]
                mb_states = states[mb_inds]
                mb_actions = actions[mb_inds]
                mb_old_log_probs = old_log_probs[mb_inds]
                mb_returns = returns_t[mb_inds]
                mb_adv = advantages_t[mb_inds]

                mean, log_std, values_pred = self.ac.forward(mb_states)
                std = torch.exp(log_std).clamp(1e-3, 1.0)
                dist = Normal(mean, std)

                new_log_prob = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                policy_loss = -(new_log_prob * mb_adv).mean()
                value_loss = F.mse_loss(values_pred, mb_returns)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.optimizer.step()

                value_losses.append(value_loss.item())
                policy_losses.append(policy_loss.item())
                entropies.append(entropy.item())
                total_loss.append(loss.item())
                

        self.get_logger().info(
            f"Update finished: policy_loss={np.mean(policy_losses):.4f} "
            f"value_loss={np.mean(value_losses):.4f} entropy={np.mean(entropies):.4f} "
            f"reward={np.mean(rewards_np)} "
            f"total_loss = {np.mean(total_loss):.4f}"
        )

        self.buffer.clear()

    def run(self):
        if not self.is_simulation_ready():
            return

        if self.stop_run_when_learning_ended():
            return

        # If previous step ended the episode, handle it BEFORE taking another step
        if self.is_episode_ended() or self.is_simulation_stopped():
            if self.buffer.has_data():
                self.finish_update()
            
            self.get_logger().info(
                f"Episode {self.episode} ended with {self.step} steps"
            )
            self.restart_learning_loop()
            self.episode += 1
            self.step = 0
            self.episode_length = 0
            self.episode_reward = 0
            self.upright_time_steps = 0
            self.prev_actions = np.zeros(self.action_size)
            self.prev_joint_velocities = np.zeros(8)
            return

        # Normal step
        self.run_one_step()

        # If the buffer is full after this step, update (on-policy)
        if self.buffer.is_full():
            self.finish_update()


def main(args=None):
    rclpy.init(args=args)
    node = ActorCriticNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
