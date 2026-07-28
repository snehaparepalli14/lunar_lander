"""Double DQN agent with the same components and settings as the DQN agent."""

from __future__ import annotations

import torch

from dqn_agent import DQNAgent


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
