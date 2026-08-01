"""Train DQN and DDQN on original and stochastic-failure LunarLander.

Example quick run (for learning and smoke testing):
    .\\.venv\\Scripts\\python.exe src\\train_experiments.py --episodes 5

Use the default 700 episodes only after confirming the quick run works in the
virtual lab. The script saves metrics and model weights under ``outputs``.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import gymnasium as gym
import numpy as np
import torch

from ddqn_agent import DDQNAgent
from dqn_agent import DQNAgent, DQNConfig
from lunar_lander_failure_env import StochasticFailureLunarLander


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


def parse_arguments() -> argparse.Namespace:
    """Read optional command-line settings for a quick or full experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        type=int,
        default=TrainingConfig.episodes,
        help="Training episodes per algorithm/environment experiment.",
    )
    return parser.parse_args()


def main() -> None:
    """Train all four required agents and save their results."""
    arguments = parse_arguments()
    if arguments.episodes <= 0:
        raise ValueError("The number of episodes must be positive.")

    training_config = TrainingConfig(episodes=arguments.episodes)
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


if __name__ == "__main__":
    main()
