"""Neural network used to estimate Q-values for LunarLander actions."""

from __future__ import annotations

import torch
from torch import nn


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
