"""Experience replay memory shared by the DQN and DDQN agents."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np


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
