"""DRL Assignment II — Single-file implementation.

Merges all source modules into one standalone script.

Usage (run from the project root  lunar_lander/):
    python drl_assignment2.py --mode verify
    python drl_assignment2.py --mode train --episodes 50
    python drl_assignment2.py --mode train
    python drl_assignment2.py --mode plot
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------
import argparse
import csv
import json
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Deque, Literal

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


# ===========================================================================
# SECTION 1 — Custom Environment Wrapper
# Source: lunar_lander_failure_env.py
# ===========================================================================

@dataclass
class FailureStatistics:
    """Counters used to verify the wrapper without exposing data to the agent."""

    attempted_thruster_actions: int = 0
    failed_thruster_actions: int = 0
    attempted_fuel_penalty: float = 0.0
    safe_landing_bonus_count: int = 0

    @property
    def observed_failure_rate(self) -> float:
        """Return the fraction of attempted thruster actions that misfired."""
        if self.attempted_thruster_actions == 0:
            return 0.0
        return self.failed_thruster_actions / self.attempted_thruster_actions


class StochasticFailureLunarLander(gym.Wrapper):
    """Apply intermittent engine failure and modified rewards to LunarLander.

    A requested thruster action (1, 2, or 3) has a 15% probability of becoming
    action 0.  The fuel penalty is based on the action requested by the agent,
    even if that action subsequently fails.
    """

    FAILURE_PROBABILITY = 0.15
    FUEL_PENALTY = 0.3
    SAFE_LANDING_BONUS = 50.0
    VELOCITY_LIMIT = 0.10
    ANGLE_LIMIT = 0.10

    def __init__(self, env: gym.Env):
        """Wrap an existing LunarLander-v3 environment."""
        super().__init__(env)
        self.statistics = FailureStatistics()

    def reset_statistics(self) -> None:
        """Clear verification counters before a new experiment."""
        self.statistics = FailureStatistics()

    def _sample_executed_action(self, selected_action: int) -> int:
        """Return the action the base environment will execute.

        Action 0 always executes.  A requested thruster action can be replaced
        by 0, but this method deliberately does not disclose that replacement
        to the learning agent.
        """
        if selected_action == 0:
            return 0

        self.statistics.attempted_thruster_actions += 1
        if self.np_random.random() < self.FAILURE_PROBABILITY:
            self.statistics.failed_thruster_actions += 1
            return 0
        return selected_action

    def _safe_landing(self, observation: np.ndarray, terminated: bool, truncated: bool) -> bool:
        """Return whether the final state meets every safe-landing condition."""
        return bool(
            terminated
            and not truncated
            and observation[6] == 1
            and observation[7] == 1
            and abs(observation[2]) < self.VELOCITY_LIMIT
            and abs(observation[3]) < self.VELOCITY_LIMIT
            and abs(observation[4]) < self.ANGLE_LIMIT
        )

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute a possibly failed action and return the specified reward.

        The returned ``info`` dictionary is exactly the base environment's
        dictionary; it contains no engine-failure or reward-modification data.
        """
        selected_action = int(action)
        if not self.action_space.contains(selected_action):
            raise ValueError(f"Invalid LunarLander action: {selected_action}")

        executed_action = self._sample_executed_action(selected_action)
        observation, base_reward, terminated, truncated, info = self.env.step(executed_action)

        fuel_penalty = self.FUEL_PENALTY if selected_action in (1, 2, 3) else 0.0
        self.statistics.attempted_fuel_penalty += fuel_penalty

        landing_bonus = 0.0
        if self._safe_landing(observation, terminated, truncated):
            landing_bonus = self.SAFE_LANDING_BONUS
            self.statistics.safe_landing_bonus_count += 1

        modified_reward = float(base_reward - fuel_penalty + landing_bonus)
        return observation, modified_reward, terminated, truncated, info


# ===========================================================================
# SECTION 2 — Q-Network Architecture
# Source: q_network.py
# ===========================================================================

class QNetwork(nn.Module):
    """Map a state vector to one predicted Q-value per discrete action.

    The network does not choose an action itself. It produces four values, and
    the agent later chooses the action with the highest value or explores.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 128,
    ) -> None:
        """Build a two-hidden-layer fully connected Q-network.

        Args:
            state_size: Number of input features; LunarLander has eight.
            action_size: Number of choices; LunarLander has four actions.
            hidden_size: Neurons in each hidden layer, shared across experiments.
        """
        super().__init__()

        # ReLU allows the network to learn non-linear landing-control patterns.
        self.layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Return Q-values with shape ``(batch_size, action_size)``.

        For one LunarLander state, the output contains predicted values for
        actions 0, 1, 2, and 3 in that order.
        """
        return self.layers(states)


# ===========================================================================
# SECTION 3 — Experience Replay Buffer
# Source: replay_buffer.py
# ===========================================================================

@dataclass(frozen=True)
class Transition:
    """One learning experience collected from a single environment step.

    Attributes:
        state: Observation before the agent selected an action.
        action: Action selected by the agent.
        reward: Modified or original reward returned by the environment.
        next_state: Observation after executing the environment step.
        done: Whether the episode ended after the step.
    """

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Store experiences and return random batches for neural-network training."""

    def __init__(self, capacity: int, seed: int) -> None:
        """Create a fixed-size memory and a reproducible random sampler.

        When the memory becomes full, ``deque`` automatically removes the
        oldest transition. This keeps memory usage bounded during training.
        """
        if capacity <= 0:
            raise ValueError("Replay-buffer capacity must be positive.")

        self.memory: Deque[Transition] = deque(maxlen=capacity)
        self.random_generator = np.random.default_rng(seed)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Copy and save one transition so later mutation cannot corrupt it."""
        # Copy arrays because Gymnasium reuses/mutates values in some settings.
        transition = Transition(
            state=np.asarray(state, dtype=np.float32).copy(),
            action=int(action),
            reward=float(reward),
            next_state=np.asarray(next_state, dtype=np.float32).copy(),
            done=bool(done),
        )
        self.memory.append(transition)

    def sample(
        self,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return a random batch as NumPy arrays ready for PyTorch tensors.

        Random sampling breaks the strong time correlation between consecutive
        simulator steps, making DQN and DDQN updates more stable.
        """
        if batch_size > len(self.memory):
            raise ValueError("Cannot sample more transitions than are stored.")

        # Sample without replacement so a batch has no duplicate transition.
        indices = self.random_generator.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[index] for index in indices]

        states = np.stack([transition.state for transition in batch])
        actions = np.asarray([transition.action for transition in batch], dtype=np.int64)
        rewards = np.asarray([transition.reward for transition in batch], dtype=np.float32)
        next_states = np.stack([transition.next_state for transition in batch])
        dones = np.asarray([transition.done for transition in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return how many transitions are currently available for sampling."""
        return len(self.memory)


# ===========================================================================
# SECTION 4 — DQN Agent
# Source: dqn_agent.py
# ===========================================================================

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


# ===========================================================================
# SECTION 5 — Double DQN Agent
# Source: ddqn_agent.py
# ===========================================================================

class DDQNAgent(DQNAgent):
    """DQN agent whose only behavioral difference is the target-Q calculation."""

    def learn(self) -> float | None:
        """Perform one DDQN update when enough replay data is available.

        The setup matches ``DQNAgent.learn``. The next action is selected by
        the online network and evaluated by the target network, avoiding DQN's
        use of one estimator for both jobs.
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

        # Both algorithms predict the value of the action actually taken.
        predicted_q_values = self.online_network(state_tensor).gather(
            dim=1,
            index=action_tensor.unsqueeze(1),
        ).squeeze(1)

        with torch.no_grad():
            # DDQN selection: online network decides which next action is best.
            next_actions = self.online_network(next_state_tensor).argmax(
                dim=1,
                keepdim=True,
            )
            # DDQN evaluation: target network scores that already selected action.
            next_q_values = self.target_network(next_state_tensor).gather(
                dim=1,
                index=next_actions,
            ).squeeze(1)
            target_q_values = reward_tensor + self.config.discount_factor * (
                1.0 - done_tensor
            ) * next_q_values

        loss = self.loss_function(predicted_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()

        # This stabilization setting is intentionally identical to DQN's.
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.learning_steps += 1
        if self.learning_steps % self.config.target_update_interval == 0:
            self.update_target_network()

        return float(loss.item())


# ===========================================================================
# SECTION 6 — Training Pipeline
# Source: train_experiments.py
# ===========================================================================

AlgorithmName = Literal["dqn", "ddqn"]
EnvironmentName = Literal["original", "modified"]
OUTPUT_DIRECTORY = Path("outputs")


@dataclass(frozen=True)
class TrainingConfig:
    """Experiment settings that remain identical across all four agents."""

    episodes: int = 700
    max_steps_per_episode: int = 1_000
    seed: int = 2026
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 500
    validation_state_count: int = 512
    success_window: int = 100


@dataclass
class TrainingMetrics:
    """Per-episode values required for the final performance plots."""

    episode_rewards: list[float]
    average_predicted_q_values: list[float]
    successful_landing_rates: list[float]
    average_thruster_activations: list[float]
    mean_losses: list[float]


def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS (Apple GPU) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_all_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS uses the CPU RNG; torch.manual_seed above covers it.


def make_environment(environment_name: EnvironmentName) -> gym.Env:
    """Create either the unchanged or stochastic-failure LunarLander version."""
    base_environment = gym.make("LunarLander-v3")
    if environment_name == "modified":
        return StochasticFailureLunarLander(base_environment)
    return base_environment


def is_safe_landing(observation: np.ndarray, terminated: bool, truncated: bool) -> bool:
    """Apply the assignment's safe-landing conditions for success-rate tracking."""
    return bool(
        terminated
        and not truncated
        and observation[6] == 1
        and observation[7] == 1
        and abs(observation[2]) < 0.10
        and abs(observation[3]) < 0.10
        and abs(observation[4]) < 0.10
    )


def epsilon_for_episode(episode: int, config: TrainingConfig) -> float:
    """Linearly reduce random exploration from epsilon_start to epsilon_end."""
    progress = min(episode / config.epsilon_decay_episodes, 1.0)
    return config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)


def collect_validation_states(config: TrainingConfig) -> np.ndarray:
    """Create one fixed state set used unchanged by all training runs.

    States are collected with a seeded random policy from original LunarLander.
    The states do not provide any reward or training information to an agent.
    """
    state_rng = np.random.default_rng(config.seed + 1)
    environment = gym.make("LunarLander-v3")
    states: list[np.ndarray] = []
    observation, _ = environment.reset(seed=config.seed + 1)

    try:
        while len(states) < config.validation_state_count:
            states.append(np.asarray(observation, dtype=np.float32).copy())
            action = int(state_rng.integers(environment.action_space.n))
            observation, _, terminated, truncated, _ = environment.step(action)
            if terminated or truncated:
                observation, _ = environment.reset()
    finally:
        environment.close()

    return np.stack(states)


def mean_max_q_value(agent: DQNAgent, validation_states: np.ndarray) -> float:
    """Return the mean best-action Q-value on the fixed validation state set."""
    state_tensor = torch.as_tensor(
        validation_states,
        dtype=torch.float32,
        device=agent.device,
    )
    with torch.no_grad():
        # The maximum is the agent's predicted value of its best action per state.
        best_action_q_values = agent.online_network(state_tensor).max(dim=1).values
    return float(best_action_q_values.mean().item())


def build_agent(
    algorithm: AlgorithmName,
    dqn_config: DQNConfig,
    seed: int,
    device: torch.device,
) -> DQNAgent:
    """Create DQN or DDQN with the same architecture and hyperparameters."""
    if algorithm == "dqn":
        return DQNAgent(dqn_config, seed, device)
    if algorithm == "ddqn":
        return DDQNAgent(dqn_config, seed, device)
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def train_one_experiment(
    algorithm: AlgorithmName,
    environment_name: EnvironmentName,
    training_config: TrainingConfig,
    dqn_config: DQNConfig,
    validation_states: np.ndarray,
) -> tuple[DQNAgent, TrainingMetrics]:
    """Train one algorithm/environment pair and collect plot-ready metrics."""
    set_all_seeds(training_config.seed)
    device = get_device()
    print(f"  Using device: {device}")
    environment = make_environment(environment_name)
    agent = build_agent(algorithm, dqn_config, training_config.seed, device)
    metrics = TrainingMetrics([], [], [], [], [])
    recent_successes: list[float] = []

    try:
        for episode in range(training_config.episodes):
            # Different deterministic reset seeds give varied but reproducible starts.
            state, _ = environment.reset(seed=training_config.seed + episode)
            episode_reward = 0.0
            episode_thruster_attempts = 0
            episode_losses: list[float] = []
            epsilon = epsilon_for_episode(episode, training_config)
            success = False

            for _ in range(training_config.max_steps_per_episode):
                action = agent.select_action(state, epsilon)
                # Count requested thruster actions consistently in both environments.
                if action in (1, 2, 3):
                    episode_thruster_attempts += 1

                next_state, reward, terminated, truncated, _ = environment.step(action)
                done = terminated or truncated
                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.learn()
                if loss is not None:
                    episode_losses.append(loss)

                episode_reward += reward
                state = next_state
                if done:
                    success = is_safe_landing(state, terminated, truncated)
                    break

            recent_successes.append(float(success))
            if len(recent_successes) > training_config.success_window:
                recent_successes.pop(0)

            metrics.episode_rewards.append(float(episode_reward))
            metrics.average_predicted_q_values.append(
                mean_max_q_value(agent, validation_states)
            )
            metrics.successful_landing_rates.append(float(np.mean(recent_successes)))
            metrics.average_thruster_activations.append(float(episode_thruster_attempts))
            metrics.mean_losses.append(
                float(np.mean(episode_losses)) if episode_losses else float("nan")
            )
    finally:
        environment.close()

    return agent, metrics


def save_experiment(
    agent: DQNAgent,
    metrics: TrainingMetrics,
    algorithm: AlgorithmName,
    environment_name: EnvironmentName,
    training_config: TrainingConfig,
    dqn_config: DQNConfig,
) -> None:
    """Save trained weights and numerical metrics for plotting and reporting."""
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    file_stem = f"{algorithm}_{environment_name}"

    torch.save(agent.online_network.state_dict(), OUTPUT_DIRECTORY / f"{file_stem}_model.pt")
    np.savez(
        OUTPUT_DIRECTORY / f"{file_stem}_metrics.npz",
        episode_rewards=np.asarray(metrics.episode_rewards),
        average_predicted_q_values=np.asarray(metrics.average_predicted_q_values),
        successful_landing_rates=np.asarray(metrics.successful_landing_rates),
        average_thruster_activations=np.asarray(metrics.average_thruster_activations),
        mean_losses=np.asarray(metrics.mean_losses),
    )
    metadata = {
        "algorithm": algorithm,
        "environment": environment_name,
        "training_config": asdict(training_config),
        "dqn_config": asdict(dqn_config),
    }
    with (OUTPUT_DIRECTORY / f"{file_stem}_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(metadata, output_file, indent=2)


def run_training(episodes: int) -> None:
    """Train all four required agents and save their results."""
    if episodes <= 0:
        raise ValueError("The number of episodes must be positive.")

    training_config = TrainingConfig(episodes=episodes)
    dqn_config = DQNConfig()
    validation_states = collect_validation_states(training_config)

    for algorithm in ("dqn", "ddqn"):
        for environment_name in ("original", "modified"):
            print(f"Training {algorithm.upper()} on {environment_name} environment...")
            agent, metrics = train_one_experiment(
                algorithm,
                environment_name,
                training_config,
                dqn_config,
                validation_states,
            )
            save_experiment(
                agent,
                metrics,
                algorithm,
                environment_name,
                training_config,
                dqn_config,
            )
            print(
                f"  final reward={metrics.episode_rewards[-1]:.1f}, "
                f"final success rate={metrics.successful_landing_rates[-1]:.1%}"
            )


# ===========================================================================
# SECTION 7 — Environment Verification
# Source: verify_environment.py
# ===========================================================================

VERIFY_SEED = 2026
VERIFY_TOTAL_STEPS = 20_000
VERIFY_FUEL_PENALTY = 0.3


def run_random_policy_verification() -> StochasticFailureLunarLander:
    """Run a seeded random policy and return the wrapper's statistics."""
    # A separate generator makes the random action sequence reproducible.
    action_rng = np.random.default_rng(VERIFY_SEED)
    environment = StochasticFailureLunarLander(gym.make("LunarLander-v3"))
    environment.reset(seed=VERIFY_SEED)

    for _ in range(VERIFY_TOTAL_STEPS):
        # Random policy: choose uniformly from the four valid discrete actions.
        selected_action = int(action_rng.integers(environment.action_space.n))
        _, _, terminated, truncated, _ = environment.step(selected_action)

        # Gymnasium requires reset before the next episode begins.
        if terminated or truncated:
            environment.reset()

    return environment


def verify_internal_statistics(environment: StochasticFailureLunarLander) -> None:
    """Assert the wrapper's aggregate fuel-penalty accounting is correct."""
    statistics = environment.statistics
    expected_penalty = statistics.attempted_thruster_actions * VERIFY_FUEL_PENALTY

    # Every thruster attempt costs fuel, including an action that misfires.
    assert np.isclose(statistics.attempted_fuel_penalty, expected_penalty)


def write_summary_csv(environment: StochasticFailureLunarLander) -> Path:
    """Save report-ready aggregate verification evidence to a CSV file."""
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "environment_verification_summary.csv"
    statistics = environment.statistics

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["metric", "value"])
        writer.writerow(["random_policy_steps", VERIFY_TOTAL_STEPS])
        writer.writerow(["attempted_thruster_actions", statistics.attempted_thruster_actions])
        writer.writerow(["failed_thruster_actions", statistics.failed_thruster_actions])
        writer.writerow(["observed_failure_rate", statistics.observed_failure_rate])
        writer.writerow(["total_fuel_penalty", statistics.attempted_fuel_penalty])
        writer.writerow(["safe_landing_bonus_count", statistics.safe_landing_bonus_count])
    return output_path


def print_verify_summary(environment: StochasticFailureLunarLander, output_path: Path) -> None:
    """Print the statistics needed for the environment-verification report."""
    statistics = environment.statistics
    print("Environment verification complete")
    print(f"Random-policy steps: {VERIFY_TOTAL_STEPS}")
    print(f"Attempted thruster actions: {statistics.attempted_thruster_actions}")
    print(f"Failed thruster actions: {statistics.failed_thruster_actions}")
    print(f"Observed failure rate: {statistics.observed_failure_rate:.2%}")
    print(f"Total attempted-action fuel penalty: {statistics.attempted_fuel_penalty:.1f}")
    print(f"Safe-landing bonuses awarded: {statistics.safe_landing_bonus_count}")
    print(f"CSV evidence written to: {output_path}")


def run_verification() -> None:
    """Run the random-policy verification and close the Box2D environment."""
    environment = run_random_policy_verification()
    try:
        verify_internal_statistics(environment)
        output_path = write_summary_csv(environment)
        print_verify_summary(environment, output_path)
    finally:
        environment.close()


# ===========================================================================
# SECTION 8 — Results Plotting
# Source: plot_results.py
# ===========================================================================

def run_plot() -> None:
    """Load saved metrics and produce the four-panel comparison figure."""
    runs = [
        ("dqn_original",  "DQN - Original",  "blue"),
        ("ddqn_original", "DDQN - Original", "cyan"),
        ("dqn_modified",  "DQN - Modified",  "red"),
        ("ddqn_modified", "DDQN - Modified", "orange"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for stem, label, color in runs:
        data = np.load(OUTPUT_DIRECTORY / f"{stem}_metrics.npz")

        axes[0, 0].plot(data["episode_rewards"],              label=label, color=color, alpha=0.6)
        axes[0, 1].plot(data["average_predicted_q_values"],   label=label, color=color)
        axes[1, 0].plot(data["successful_landing_rates"],     label=label, color=color)
        axes[1, 1].plot(data["average_thruster_activations"], label=label, color=color, alpha=0.6)

    titles = [
        "1. Episode Reward vs Episode",
        "2. Avg Predicted Q-value (Validation Set)",
        "3. Landing Success Rate (100-Ep Moving Avg)",
        "4. Avg Thruster Activations per Episode",
    ]

    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    plt.tight_layout()
    out_path = OUTPUT_DIRECTORY / "drl_performance_comparison.png"
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved to: {out_path}")
    plt.show()


# ===========================================================================
# SECTION 9 — Unified Command-Line Entry Point
# ===========================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse the unified CLI for all three modes."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["verify", "train", "plot"],
        default="train",
        help=(
            "verify  — run environment wrapper verification\n"
            "train   — train all four DQN/DDQN agents (default)\n"
            "plot    — generate comparison plots from saved outputs"
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=TrainingConfig.episodes,
        help="Training episodes per experiment (only used with --mode train).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.mode == "verify":
        run_verification()
    elif args.mode == "train":
        run_training(args.episodes)
    elif args.mode == "plot":
        run_plot()
