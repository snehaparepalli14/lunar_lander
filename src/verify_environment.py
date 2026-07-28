"""Verify the stochastic-action wrapper with a reproducible random-policy run.

Run from the project root:
    .\\.venv\\Scripts\\python.exe src\\verify_environment.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import gymnasium as gym
import numpy as np

from lunar_lander_failure_env import StochasticFailureLunarLander


SEED = 2026
TOTAL_STEPS = 20_000
FUEL_PENALTY = 0.3
OUTPUT_DIRECTORY = Path("outputs")


def run_random_policy_verification() -> StochasticFailureLunarLander:
    """Run a seeded random policy and return the wrapper's statistics."""
    # A separate generator makes the random action sequence reproducible.
    action_rng = np.random.default_rng(SEED)
    environment = StochasticFailureLunarLander(gym.make("LunarLander-v3"))
    environment.reset(seed=SEED)

    for _ in range(TOTAL_STEPS):
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
    expected_penalty = statistics.attempted_thruster_actions * FUEL_PENALTY

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
        writer.writerow(["random_policy_steps", TOTAL_STEPS])
        writer.writerow(["attempted_thruster_actions", statistics.attempted_thruster_actions])
        writer.writerow(["failed_thruster_actions", statistics.failed_thruster_actions])
        writer.writerow(["observed_failure_rate", statistics.observed_failure_rate])
        writer.writerow(["total_fuel_penalty", statistics.attempted_fuel_penalty])
        writer.writerow(["safe_landing_bonus_count", statistics.safe_landing_bonus_count])
    return output_path


def print_summary(environment: StochasticFailureLunarLander, output_path: Path) -> None:
    """Print the statistics needed for the environment-verification report."""
    statistics = environment.statistics
    print("Environment verification complete")
    print(f"Random-policy steps: {TOTAL_STEPS}")
    print(f"Attempted thruster actions: {statistics.attempted_thruster_actions}")
    print(f"Failed thruster actions: {statistics.failed_thruster_actions}")
    print(f"Observed failure rate: {statistics.observed_failure_rate:.2%}")
    print(f"Total attempted-action fuel penalty: {statistics.attempted_fuel_penalty:.1f}")
    print(f"Safe-landing bonuses awarded: {statistics.safe_landing_bonus_count}")
    print(f"CSV evidence written to: {output_path}")


def main() -> None:
    """Run the random-policy verification and close the Box2D environment."""
    environment = run_random_policy_verification()
    try:
        verify_internal_statistics(environment)
        output_path = write_summary_csv(environment)
        print_summary(environment, output_path)
    finally:
        environment.close()


if __name__ == "__main__":
    main()
