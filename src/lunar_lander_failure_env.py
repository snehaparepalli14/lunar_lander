"""Custom LunarLander environment with stochastic engine failures.

The wrapper preserves LunarLander-v3's observation/action spaces and episode
rules.  It changes only the executed action and the reward, as required by the
assignment specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np


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
