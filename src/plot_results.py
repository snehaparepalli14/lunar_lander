"""Create the four performance plots required by the LunarLander assignment.

Run after all four training experiments have completed:
    .\\.venv\\Scripts\\python.exe src\\plot_results.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIRECTORY = Path("outputs")
PLOT_DIRECTORY = OUTPUT_DIRECTORY / "plots"
EXPERIMENTS = {
    "DQN - Original": "dqn_original",
    "DDQN - Original": "ddqn_original",
    "DQN - Modified": "dqn_modified",
    "DDQN - Modified": "ddqn_modified",
}
COLORS = {
    "DQN - Original": "#1f77b4",
    "DDQN - Original": "#ff7f0e",
    "DQN - Modified": "#2ca02c",
    "DDQN - Modified": "#d62728",
}


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Return a trailing moving average while keeping the input length unchanged."""
    if window <= 0:
        raise ValueError("Moving-average window must be positive.")

    # Use smaller windows at the beginning so every episode still has a value.
    cumulative_sum = np.cumsum(np.insert(values.astype(float), 0, 0.0))
    averages = np.empty(len(values), dtype=float)
    for index in range(len(values)):
        start_index = max(0, index - window + 1)
        count = index - start_index + 1
        averages[index] = (
            cumulative_sum[index + 1] - cumulative_sum[start_index]
        ) / count
    return averages


def load_metrics() -> dict[str, dict[str, np.ndarray]]:
    """Load the metrics saved by all four completed training experiments."""
    loaded_metrics: dict[str, dict[str, np.ndarray]] = {}
    for display_name, file_stem in EXPERIMENTS.items():
        metrics_path = OUTPUT_DIRECTORY / f"{file_stem}_metrics.npz"
        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Missing {metrics_path}. Run all four training experiments first."
            )
        data = np.load(metrics_path)
        loaded_metrics[display_name] = {
            key: data[key]
            for key in data.files
        }
    return loaded_metrics


def plot_metric(
    metrics: dict[str, dict[str, np.ndarray]],
    metric_key: str,
    title: str,
    y_label: str,
    output_name: str,
    smoothing_window: int | None = None,
    percentage: bool = False,
) -> None:
    """Create one consistently styled comparison chart for a saved metric."""
    figure, axis = plt.subplots(figsize=(11, 6))

    for display_name, experiment_metrics in metrics.items():
        values = experiment_metrics[metric_key]
        episodes = np.arange(1, len(values) + 1)
        color = COLORS[display_name]

        if smoothing_window is None:
            axis.plot(episodes, values, label=display_name, color=color, linewidth=2)
        else:
            # Faint raw values retain evidence; the average makes the trend readable.
            axis.plot(episodes, values, color=color, alpha=0.15, linewidth=0.8)
            axis.plot(
                episodes,
                moving_average(values, smoothing_window),
                label=f"{display_name} ({smoothing_window}-episode average)",
                color=color,
                linewidth=2,
            )

    axis.set_title(title)
    axis.set_xlabel("Training episode")
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=9)
    if percentage:
        axis.set_ylim(0.0, 1.0)
        axis.yaxis.set_major_formatter("{x:.0%}")

    figure.tight_layout()
    figure.savefig(PLOT_DIRECTORY / output_name, dpi=200)
    plt.close(figure)


def main() -> None:
    """Load training results and save all four assignment-required plots."""
    PLOT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()

    plot_metric(
        metrics,
        metric_key="episode_rewards",
        title="Episode Reward versus Training Episode",
        y_label="Episode reward",
        output_name="episode_rewards.png",
        smoothing_window=25,
    )
    plot_metric(
        metrics,
        metric_key="average_predicted_q_values",
        title="Average Predicted Q-value versus Training Episode",
        y_label="Mean best-action predicted Q-value",
        output_name="average_predicted_q_values.png",
        smoothing_window=25,
    )
    plot_metric(
        metrics,
        metric_key="successful_landing_rates",
        title="Safe Landing Rate versus Training Episode",
        y_label="Safe landing rate (previous 100 episodes)",
        output_name="safe_landing_rates.png",
        percentage=True,
    )
    plot_metric(
        metrics,
        metric_key="average_thruster_activations",
        title="Average Thruster Activations versus Training Episode",
        y_label="Requested thruster actions per episode",
        output_name="thruster_activations.png",
        smoothing_window=25,
    )
    print(f"Saved four plots to: {PLOT_DIRECTORY}")


if __name__ == "__main__":
    main()
