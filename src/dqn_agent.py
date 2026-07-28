"""Deep Q-Network agent for the LunarLander experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from q_network import QNetwork
from replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class DQNConfig:
    """Hyperparameters shared by DQN and DDQN for a fair experiment."""

    state_size: int = 8
    action_size: int = 4
    hidden_size: int = 128
    replay_capacity: int = 100_000
    batch_size: int = 64
    learning_rate: float = 1e-3
    discount_factor: float = 0.99
    target_update_interval: int = 1_000


class DQNAgent:
    """Learn action values with experience replay and a target network."""

    def __init__(self, config: DQNConfig, seed: int, device: torch.device) -> None:
        """Create online/target networks, optimizer, and replay memory.

        The online network learns every update. The target network is a delayed
        copy used only to calculate stable learning targets.
        """
        self.config = config
        self.device = device
        self.random_generator = np.random.default_rng(seed)

        # Both networks start identically; only the online network is optimized.
        self.online_network = QNetwork(
            config.state_size,
            config.action_size,
            config.hidden_size,
        ).to(device)
        self.target_network = QNetwork(
            config.state_size,
            config.action_size,
            config.hidden_size,
        ).to(device)
        self.update_target_network()
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=config.learning_rate,
        )
        self.loss_function = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(config.replay_capacity, seed)
        self.learning_steps = 0

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """Choose a random action with probability epsilon; otherwise exploit.

        Epsilon-greedy exploration lets the agent discover useful actions before
        its Q-value estimates become reliable.
        """
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("Epsilon must be between 0 and 1.")

        # Explore by sampling one of the valid actions uniformly at random.
        if self.random_generator.random() < epsilon:
            return int(self.random_generator.integers(self.config.action_size))

        # Disable gradients because action selection does not train the network.
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = self.online_network(state_tensor.unsqueeze(0))
        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Save a completed environment interaction in replay memory."""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def learn(self) -> float | None:
        """Perform one gradient update when enough experiences are available.

        Returns:
            The scalar loss after an update, or ``None`` before the replay
            buffer contains one complete batch.
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.batch_size
        )
        state_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_state_tensor = torch.as_tensor(
            next_states,
            dtype=torch.float32,
            device=self.device,
        )
        done_tensor = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # Select the online-network value for the action actually taken.
        predicted_q_values = self.online_network(state_tensor).gather(
            dim=1,
            index=action_tensor.unsqueeze(1),
        ).squeeze(1)

        # DQN evaluates the largest next-state Q-value using the target network.
        with torch.no_grad():
            next_q_values = self.target_network(next_state_tensor).max(dim=1).values
            target_q_values = reward_tensor + self.config.discount_factor * (
                1.0 - done_tensor
            ) * next_q_values

        loss = self.loss_function(predicted_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping prevents a rare large update from destabilizing learning.
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.learning_steps += 1
        if self.learning_steps % self.config.target_update_interval == 0:
            self.update_target_network()

        return float(loss.item())

    def update_target_network(self) -> None:
        """Copy online-network weights to the delayed target network."""
        self.target_network.load_state_dict(self.online_network.state_dict())

    def mean_q_value(self, states: np.ndarray) -> float:
        """Return mean predicted Q-value over a fixed validation-state set."""
        state_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = self.online_network(state_tensor)
        return float(q_values.mean().item())
