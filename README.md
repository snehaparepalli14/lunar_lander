# Robust Reinforcement Learning under Stochastic Action Failure

This project implements the Assignment II comparison of Deep Q-Network (DQN)
and Double Deep Q-Network (DDQN) agents on two Gymnasium environments:

1. The original `LunarLander-v3` environment.
2. A custom wrapper where requested thruster actions fail with 15% probability.

The code is written as a learning project. Each function has a docstring and
important reinforcement-learning decisions have inline comments.

## Assignment goal

The experiment answers this question:

> When engine commands fail unpredictably, does DDQN learn more reliable action
> values and landing behaviour than ordinary DQN?

Four trained agents are compared:

| Algorithm | Environment |
| --- | --- |
| DQN | Original LunarLander-v3 |
| DDQN | Original LunarLander-v3 |
| DQN | Stochastic action-failure wrapper |
| DDQN | Stochastic action-failure wrapper |

## Project structure

```text
D:\\DRL
├── src/
│   ├── lunar_lander_failure_env.py  # Custom Gymnasium wrapper
│   ├── verify_environment.py        # Random-policy verification
│   ├── replay_buffer.py             # Experience replay memory
│   ├── q_network.py                 # Shared neural-network architecture
│   ├── dqn_agent.py                 # DQN agent
│   ├── ddqn_agent.py                # DDQN agent
│   └── train_experiments.py         # Four-experiment training pipeline
├── outputs/                         # Generated logs, metrics, and models
├── requirements.txt                 # Python dependencies
├── architecture.md                  # Detailed technical reference
└── README.md                        # This file
```

## Environment modification

The custom wrapper changes only action execution and reward:

- Action `0` (do nothing) is executed normally.
- A requested action `1`, `2`, or `3` becomes action `0` with probability `0.15`.
- The failure is not added to the returned `info` dictionary.
- Fuel penalty is `0.3` for every requested thruster action, even if it misfires.
- A `+50` bonus is awarded only for the specified safe landing: normal
  termination, both legs in contact, low velocity, and near-level orientation.

The reward is:

```text
modified reward = base reward - fuel penalty + safe landing bonus
```

## Setup

The project uses a local virtual environment. Dependencies are installed in
`.venv`, not into system Python.

From PowerShell in `D:\\DRL`:

```powershell
.\\.venv\\Scripts\\Activate.ps1
```

If the environment has not been created yet:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
```

Alternatively, use the isolated Python executable directly:

```powershell
.\\.venv\\Scripts\\python.exe --version
```

## Run the environment verification

This executes a seeded random policy for 20,000 steps. It verifies that the
observed failure rate is close to 15% and checks fuel-penalty accounting.

```powershell
.\\.venv\\Scripts\\python.exe src\\verify_environment.py
```

The summary CSV is saved as `outputs/environment_verification_summary.csv`.
Include the console output and relevant CSV evidence in the report.

## Run training

First use a short run to confirm the full pipeline works:

```powershell
.\\.venv\\Scripts\\python.exe src\\train_experiments.py --episodes 50
```

Then run the configured full experiment:

```powershell
.\\.venv\\Scripts\\python.exe src\\train_experiments.py
```

The default is 700 episodes for each of the four experiment combinations. The
run saves a model (`.pt`), metrics (`.npz`), and settings (`.json`) under
`outputs/` for each combination.

## Interpreting the saved metrics

| Metric | Meaning |
| --- | --- |
| `episode_rewards` | Total modified/original reward in each episode. |
| `average_predicted_q_values` | Mean best-action Q-value over fixed validation states. |
| `successful_landing_rates` | Moving average of safe landings over 100 episodes. |
| `average_thruster_activations` | Number of requested thruster actions per episode. |
| `mean_losses` | Mean neural-network loss for updates in an episode. |

## Fair-comparison rules

Do not change DQN and DDQN independently. The assignment requires identical
seed, architecture, optimizer, replay buffer, exploration schedule, training
duration, and target-network update schedule.

The only algorithmic difference is target-Q calculation:

```text
DQN:  target network selects and evaluates the best next action.
DDQN: online network selects; target network evaluates that action.
```

## Submission reminders

- Submit one PDF with commented code and outputs.
- Add group members and contribution percentages at the beginning.
- Execute final work in the required virtual lab and include timestamped
  screenshots.
- Explain the experiment in your own words before submitting.
