#!/usr/bin/env python3
"""Generate publication-quality plots from experiment results.

Usage::

    python -m arithmetic_llm.plot_results experiments/run_01

Reads ``consolidated_results.json`` and per-condition training logs,
then produces the following figures:

1. **Training dynamics** — reward rate, loss, KL over steps (all conditions)
2. **Reasoning quality taxonomy** — stacked bar chart
3. **Error-type distribution** — grouped bars
4. **Per-depth accuracy** — line plot
5. **Reward-accuracy divergence** — reward vs. true accuracy over time
6. **Component score evolution** — format/process/consistency/outcome over steps
7. **Phase timeline** — for the scheduled condition
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

# Lazy import matplotlib so the module can be imported without it installed
_MPL_AVAILABLE = True
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    _MPL_AVAILABLE = False


# Consistent colors per condition
CONDITION_COLORS = {
    "outcome_only": "#1f77b4",
    "naive_process": "#ff7f0e",
    "error_aware": "#2ca02c",
    "scheduled": "#d62728",
}

CONDITION_LABELS = {
    "outcome_only": "Outcome Only",
    "naive_process": "Naive Process",
    "error_aware": "Error Aware",
    "scheduled": "Scheduled",
}

TAXONOMY_COLORS = {
    "correct_reasoning": "#2ca02c",
    "specious_cot": "#ff7f0e",
    "unlucky": "#9467bd",
    "failed": "#d62728",
}

TAXONOMY_LABELS = {
    "correct_reasoning": "Correct Reasoning",
    "specious_cot": "Specious CoT",
    "unlucky": "Unlucky",
    "failed": "Failed",
}


def _require_matplotlib():
    if not _MPL_AVAILABLE:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        )


def _ema(values: List[float], alpha: float = 0.1) -> List[float]:
    """Exponential moving average smoothing (alpha = smoothing factor, lower = smoother)."""
    if not values:
        return values
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    return smoothed


def load_training_log(cond_dir: str) -> List[dict]:
    """Load the GRPO training log JSON from a condition directory."""
    # The training directory may be nested (grpo_YYYYMMDD_...)
    log_path = os.path.join(cond_dir, "grpo_training_log.json")
    if os.path.exists(log_path):
        with open(log_path) as f:
            return json.load(f)

    # Search subdirectories
    for entry in os.listdir(cond_dir):
        candidate = os.path.join(cond_dir, entry, "grpo_training_log.json")
        if os.path.exists(candidate):
            with open(candidate) as f:
                return json.load(f)
    return []


def load_consolidated(experiment_dir: str) -> dict:
    path = os.path.join(experiment_dir, "consolidated_results.json")
    with open(path) as f:
        return json.load(f)



def plot_training_dynamics(experiment_dir: str, conditions: List[str]) -> None:
    _require_matplotlib()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    metrics_keys = ["reward_rate", "total_loss", "kl_divergence"]
    titles = ["Reward Rate", "Total Loss", "KL Divergence"]

    for cond in conditions:
        cond_dir = os.path.join(experiment_dir, cond)
        log = load_training_log(cond_dir)
        if not log:
            continue

        steps = [e["step"] for e in log if e.get("step", 0) > 0]
        for ax, key, title in zip(axes, metrics_keys, titles):
            values = [
                e["metrics"].get(key, 0) for e in log if e.get("step", 0) > 0
            ]
            if values:
                color = CONDITION_COLORS.get(cond, None)
                smoothed = _ema(values, alpha=0.05)
                # Raw values as faint background
                ax.plot(
                    steps[:len(values)], values,
                    color=color, alpha=0.15, linewidth=0.8,
                )
                # Smoothed line in foreground
                ax.plot(
                    steps[:len(smoothed)], smoothed,
                    label=CONDITION_LABELS.get(cond, cond),
                    color=color, alpha=0.9, linewidth=1.8,
                )

    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xlabel("Training Step")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        os.path.join(experiment_dir, "plot_training_dynamics.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print("  Saved: plot_training_dynamics.png")



def plot_taxonomy(experiment_dir: str, eval_results: dict) -> None:
    _require_matplotlib()

    conditions = list(eval_results.keys())
    taxonomy_keys = ["correct_reasoning", "specious_cot", "unlucky", "failed"]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = range(len(conditions))
    bottoms = [0.0] * len(conditions)

    for tk in taxonomy_keys:
        values = [
            eval_results[c].get("taxonomy_fractions", {}).get(tk, 0)
            for c in conditions
        ]
        ax.bar(
            x, values, bottom=bottoms,
            label=TAXONOMY_LABELS.get(tk, tk),
            color=TAXONOMY_COLORS.get(tk, None),
        )
        bottoms = [b + v for b, v in zip(bottoms, values)]

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conditions], rotation=15)
    ax.set_ylabel("Fraction of Samples")
    ax.set_title("Reasoning Quality Taxonomy")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(
        os.path.join(experiment_dir, "plot_taxonomy.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print("  Saved: plot_taxonomy.png")



def plot_error_distribution(experiment_dir: str, eval_results: dict) -> None:
    _require_matplotlib()

    conditions = list(eval_results.keys())
    error_types = ["independent", "propagated", "invalid_operands"]
    error_labels = ["Independent", "Propagated", "Invalid Operands"]
    error_colors = ["#d62728", "#ff7f0e", "#9467bd"]

    fig, ax = plt.subplots(figsize=(8, 5))

    bar_width = 0.2
    x_base = list(range(len(conditions)))

    for i, (etype, elabel, ecolor) in enumerate(zip(error_types, error_labels, error_colors)):
        values = [
            eval_results[c].get("error_distribution", {}).get(etype, 0)
            for c in conditions
        ]
        offsets = [xb + i * bar_width for xb in x_base]
        ax.bar(offsets, values, width=bar_width, label=elabel, color=ecolor)

    center_offsets = [xb + bar_width for xb in x_base]
    ax.set_xticks(center_offsets)
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conditions], rotation=15)
    ax.set_ylabel("Total Error Count")
    ax.set_title("Error-Type Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(
        os.path.join(experiment_dir, "plot_error_distribution.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print("  Saved: plot_error_distribution.png")



def plot_per_depth_accuracy(experiment_dir: str, eval_results: dict) -> None:
    _require_matplotlib()

    fig, ax = plt.subplots(figsize=(8, 5))

    for cond, data in eval_results.items():
        raw = data.get("per_depth", {})
        if not raw:
            continue
        # Normalize keys to int — int in-memory, str when loaded from JSON
        per_depth = {int(k): v for k, v in raw.items()}
        depths = sorted(per_depth.keys())
        accs = [per_depth[d]["accuracy"] for d in depths]
        ax.plot(
            depths, accs,
            marker="o",
            label=CONDITION_LABELS.get(cond, cond),
            color=CONDITION_COLORS.get(cond, None),
        )

    ax.set_xlabel("Expression Depth")
    ax.set_ylabel("Exact Match Accuracy")
    ax.set_title("Accuracy by Expression Depth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(
        os.path.join(experiment_dir, "plot_per_depth_accuracy.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print("  Saved: plot_per_depth_accuracy.png")



def plot_reward_accuracy_divergence(experiment_dir: str, conditions: List[str]) -> None:
    """Two-panel plot: composite training reward and true outcome score over time.

    Panel 1 shows the composite reward each condition optimises.
    Panel 2 shows the true outcome score (binary accuracy proxy) from the same
    training log — a gap between the two panels indicates reward hacking.
    Both series are EMA-smoothed so trends are visible.
    """
    _require_matplotlib()

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={"hspace": 0.08},
    )

    for cond in conditions:
        cond_dir = os.path.join(experiment_dir, cond)
        log = load_training_log(cond_dir)
        if not log:
            continue

        entries = [e for e in log if e.get("step", 0) > 0]
        steps = [e["step"] for e in entries]
        color = CONDITION_COLORS.get(cond, None)
        label = CONDITION_LABELS.get(cond, cond)

        # Composite training reward (what the model is actually optimised on)
        rewards = [e["metrics"].get("avg_reward", 0) for e in entries]
        # Outcome score — binary correctness, logged as avg_outcome_score when
        # using process rewards, or reward_rate for outcome_only
        outcome_scores = [
            e["metrics"].get(
                "avg_outcome_score",
                e["metrics"].get("reward_rate", 0),
            )
            for e in entries
        ]

        if not rewards:
            continue

        sm_rewards = _ema(rewards, alpha=0.05)
        sm_outcome = _ema(outcome_scores, alpha=0.05)

        # Raw faint traces
        ax_top.plot(steps, rewards, color=color, alpha=0.12, linewidth=0.8)
        ax_bot.plot(steps, outcome_scores, color=color, alpha=0.12, linewidth=0.8)
        # Smoothed foreground traces
        ax_top.plot(steps, sm_rewards, label=label, color=color,
                    alpha=0.9, linewidth=1.8)
        ax_bot.plot(steps, sm_outcome, label=label, color=color,
                    alpha=0.9, linewidth=1.8)

    ax_top.set_ylabel("Composite Training Reward")
    ax_top.set_title(
        "Reward vs. Outcome Score Over Training\n"
        "(gap between panels = non-outcome components driving the reward)"
    )
    ax_top.set_ylim(-0.05, 1.05)
    ax_top.legend(fontsize=8)
    ax_top.grid(True, alpha=0.3)

    ax_bot.set_ylabel("True Outcome Score")
    ax_bot.set_xlabel("Training Step")
    ax_bot.set_ylim(-0.05, 1.05)
    ax_bot.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        os.path.join(experiment_dir, "plot_reward_divergence.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print("  Saved: plot_reward_divergence.png")



def plot_component_evolution(experiment_dir: str, conditions: List[str]) -> None:
    _require_matplotlib()

    component_keys = [
        "avg_format_score", "avg_process_score",
        "avg_consistency_score", "avg_outcome_score",
    ]
    component_labels = ["Format", "Process", "Consistency", "Outcome"]

    # Only plot conditions that have component scores
    valid_conditions = []
    for cond in conditions:
        cond_dir = os.path.join(experiment_dir, cond)
        log = load_training_log(cond_dir)
        if any("avg_format_score" in e.get("metrics", {}) for e in log):
            valid_conditions.append(cond)

    if not valid_conditions:
        print("  Skipping component evolution plot (no component scores logged)")
        return

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for cond in valid_conditions:
        cond_dir = os.path.join(experiment_dir, cond)
        log = load_training_log(cond_dir)
        entries = [
            e for e in log
            if e.get("step", 0) > 0 and "avg_format_score" in e.get("metrics", {})
        ]
        steps = [e["step"] for e in entries]

        for ax, key, label in zip(axes, component_keys, component_labels):
            values = [e["metrics"].get(key, 0) for e in entries]
            if values:
                color = CONDITION_COLORS.get(cond, None)
                smoothed = _ema(values, alpha=0.05)
                ax.plot(
                    steps[:len(values)], values,
                    color=color, alpha=0.12, linewidth=0.8,
                )
                ax.plot(
                    steps[:len(smoothed)], smoothed,
                    label=CONDITION_LABELS.get(cond, cond),
                    color=color, alpha=0.9, linewidth=1.8,
                )

    for ax, label in zip(axes, component_labels):
        ax.set_title(f"{label} Score")
        ax.set_xlabel("Step")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        os.path.join(experiment_dir, "plot_component_evolution.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print("  Saved: plot_component_evolution.png")



def plot_phase_timeline(experiment_dir: str) -> None:
    _require_matplotlib()

    cond_dir = os.path.join(experiment_dir, "scheduled")
    log = load_training_log(cond_dir)
    if not log:
        print("  Skipping phase timeline (no scheduled condition data)")
        return

    entries = [
        e for e in log
        if e.get("step", 0) > 0 and "reward_phase" in e.get("metrics", {})
    ]
    if not entries:
        print("  Skipping phase timeline (no phase data)")
        return

    steps = [e["step"] for e in entries]
    phases = [e["metrics"]["reward_phase"] for e in entries]

    # Also plot reward weights if available
    has_weights = "reward_weights" in entries[0].get("metrics", {})

    if has_weights:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=(10, 3))

    # Phase number
    ax1.plot(steps, phases, color="#d62728", linewidth=2)
    ax1.set_ylabel("Phase")
    ax1.set_yticks([1, 2, 3])
    ax1.set_yticklabels(["Format", "Process", "Outcome"])
    ax1.set_title("Adaptive Reward Schedule Phase Timeline")
    ax1.grid(True, alpha=0.3)

    if has_weights:
        weight_labels = ["Format", "Process", "Consistency", "Outcome"]
        weight_colors = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728"]
        for i, (wlabel, wcolor) in enumerate(zip(weight_labels, weight_colors)):
            values = [e["metrics"]["reward_weights"][i] for e in entries]
            ax2.plot(steps[:len(values)], values, label=wlabel, color=wcolor)
        ax2.set_xlabel("Training Step")
        ax2.set_ylabel("Weight")
        ax2.set_title("Reward Weight Evolution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        os.path.join(experiment_dir, "plot_phase_timeline.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print("  Saved: plot_phase_timeline.png")



def generate_all_plots(experiment_dir: str) -> None:
    """Generate all plots from experiment results."""
    _require_matplotlib()

    print(f"\nGenerating plots from {experiment_dir}")
    print("=" * 50)

    consolidated = load_consolidated(experiment_dir)
    conditions = consolidated.get("conditions", [])
    eval_results = consolidated.get("evaluation_results", {})

    plot_training_dynamics(experiment_dir, conditions)
    if eval_results:
        plot_taxonomy(experiment_dir, eval_results)
        plot_error_distribution(experiment_dir, eval_results)
        plot_per_depth_accuracy(experiment_dir, eval_results)
    plot_reward_accuracy_divergence(experiment_dir, conditions)
    plot_component_evolution(experiment_dir, conditions)
    plot_phase_timeline(experiment_dir)

    print(f"\nAll plots saved to {experiment_dir}")



def main() -> None:
    parser = argparse.ArgumentParser(description="Generate experiment plots")
    parser.add_argument(
        "experiment_dir",
        type=str,
        help="Path to experiment output directory",
    )
    args = parser.parse_args()
    generate_all_plots(args.experiment_dir)


if __name__ == "__main__":
    main()
