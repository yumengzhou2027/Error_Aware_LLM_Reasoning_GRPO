#!/usr/bin/env python3
"""Re-evaluate trained GRPO checkpoints on OOD test sets without retraining.

Run this on the same machine that holds the experiment checkpoints.
Writes new evaluation results to the experiment directory, which you can
then copy back and re-plot.

Usage::

    python -m arithmetic_llm.reeval_ood \
        --experiment-dir experiments/run_01 \
        --tokenizer     data/tokenizer \
        --eval-samples  800

This produces five evaluation passes per condition:

  1. in_dist      : depth 1-5,   range 1-20    (matches training distribution)
  2. depth_sweep  : depth 1-8,   range 1-20    (fine-grained depth curve, 8 levels)
  3. ood_range    : depth 1-5,   range 1-1000  (large numbers, familiar depth)
  4. ood_hard     : depth 6-8,   range 1-1000  (deeper + large numbers)
  5. ood_extreme  : depth 8,     range 1-1000  (hardest single-depth bucket)

The model was trained on depth 1-5 and range 1-20, so anything beyond that is
out-of-distribution.  Depth 8 with range 1-1000 produces solutions of ~350
tokens, safely within the model's 512-token positional embedding limit.

Results are saved to
  <experiment-dir>/ood_results.json          (all conditions, all test sets)
  <experiment-dir>/plot_ood_*.png            (comparison plots)

and the consolidated_results.json is updated with the new ood_results key.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch

from .arithmetic_tokenizer import ArithmeticBPETokenizer
from .evaluator import ModelEvaluator
from .extended_evaluator import ExtendedEvaluator, EvaluationReport, save_evaluation_report


# ──────────────────────────────────────────────
# Test-set configurations
# ──────────────────────────────────────────────

OOD_SETS: Dict[str, dict] = {
    "in_dist": {
        "label": "In-Dist (d1-5, r1-20)",
        "depths": [1, 2, 3, 4, 5],
        "num_range": (1, 20),
        "bar_chart": True,   # include in grouped bar chart
    },
    "depth_sweep": {
        # Fine-grained depth curve: one bucket per depth from 1 to 8.
        # Used for the accuracy-by-depth line plot only.
        "label": "Depth Sweep (d1-8, r1-20)",
        "depths": [1, 2, 3, 4, 5, 6, 7, 8],
        "num_range": (1, 20),
        "bar_chart": False,  # too many depths to show as a bar; use depth curve instead
    },
    "ood_range": {
        "label": "OOD Range (d1-5, r1-1000)",
        "depths": [1, 2, 3, 4, 5],
        "num_range": (1, 1000),
        "bar_chart": True,
    },
    "ood_hard": {
        "label": "OOD Hard (d6-8, r1-1000)",
        "depths": [6, 7, 8],
        "num_range": (1, 1000),
        "bar_chart": True,
    },
    "ood_extreme": {
        # Hardest single bucket: all samples at depth 8, large numbers.
        "label": "OOD Extreme (d8, r1-1000)",
        "depths": [8],
        "num_range": (1, 1000),
        "bar_chart": True,
    },
}

CONDITION_LABELS = {
    "outcome_only": "Outcome Only",
    "naive_process": "Naive Process",
    "error_aware":   "Error Aware",
    "scheduled":     "Scheduled",
}

CONDITION_COLORS = {
    "outcome_only": "#1f77b4",
    "naive_process": "#ff7f0e",
    "error_aware":   "#2ca02c",
    "scheduled":     "#d62728",
}


# ──────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────

def _find_checkpoint(cond_dir: str) -> Optional[str]:
    """Return path to best available checkpoint in a condition directory."""
    final = os.path.join(cond_dir, "final_model.pt")
    if os.path.exists(final):
        return final
    import glob as _glob
    ckpts = sorted(_glob.glob(os.path.join(cond_dir, "checkpoint_step_*.pt")))
    if ckpts:
        return ckpts[-1]
    # Also search one level deeper (grpo_YYYYMMDD subdirs)
    for entry in os.listdir(cond_dir):
        sub = os.path.join(cond_dir, entry)
        if os.path.isdir(sub):
            final_sub = os.path.join(sub, "final_model.pt")
            if os.path.exists(final_sub):
                return final_sub
    return None


def evaluate_one_set(
    checkpoint_path: str,
    tokenizer_path: str,
    depths: List[int],
    num_range: Tuple[int, int],
    num_samples: int,
    seed: int = 456,
) -> EvaluationReport:
    """Load checkpoint, generate, and run extended evaluation."""
    evaluator_model = ModelEvaluator(
        model_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
    )

    def generate_fn(prompt: str) -> str:
        return evaluator_model._generate_solution(prompt, max_length=512)

    ext_eval = ExtendedEvaluator(generate_fn)
    report = ext_eval.evaluate(
        num_samples=num_samples,
        max_depth=max(depths),
        num_range=num_range,
        depths=depths,
        seed=seed,
    )
    return report


# ──────────────────────────────────────────────
# Plotting (self-contained so this script is runnable standalone)
# ──────────────────────────────────────────────

def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("pip install matplotlib")


def plot_ood_accuracy_bars(
    results: Dict[str, Dict[str, dict]],
    output_dir: str,
) -> None:
    """Grouped bar chart: accuracy for each condition × test set."""
    plt = _require_matplotlib()
    import matplotlib.pyplot as mpl_plt
    import numpy as np

    conditions = list(results.keys())
    # Only include test sets flagged for bar chart (excludes depth_sweep)
    test_sets = [ts for ts, cfg in OOD_SETS.items() if cfg.get("bar_chart", True)]
    labels = [OOD_SETS[ts]["label"] for ts in test_sets]

    x = np.arange(len(test_sets))
    width = 0.2
    offsets = np.linspace(
        -(len(conditions) - 1) * width / 2,
        (len(conditions) - 1) * width / 2,
        len(conditions),
    )

    fig, ax = mpl_plt.subplots(figsize=(11, 5))

    for cond, offset in zip(conditions, offsets):
        accs = [
            results[cond].get(ts, {}).get("exact_match_accuracy", 0.0)
            for ts in test_sets
        ]
        ax.bar(
            x + offset, accs, width,
            label=CONDITION_LABELS.get(cond, cond),
            color=CONDITION_COLORS.get(cond, None),
            alpha=0.85,
            edgecolor="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Exact Match Accuracy")
    ax.set_title("OOD Generalization: Accuracy by Test Set and Condition")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    path = os.path.join(output_dir, "plot_ood_accuracy_bars.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    mpl_plt.close(fig)
    print(f"  Saved: {path}")


def plot_ood_depth_curves(
    results: Dict[str, Dict[str, dict]],
    output_dir: str,
) -> None:
    """Two-panel line plot: accuracy vs depth for all conditions.

    Left panel  — depth_sweep (depths 1-8, range 1-20): isolates the effect of
                  increasing expression depth on a familiar number range.
    Right panel — ood_hard (depths 6-8, range 1-1000): the fully OOD regime.
    Both panels share the same conditions and color scheme.
    """
    plt = _require_matplotlib()
    import matplotlib.pyplot as mpl_plt
    import matplotlib.ticker as ticker

    panels = [
        ("depth_sweep", "Depth Sweep — range 1-20\n(isolates depth effect)"),
        ("ood_hard",    "OOD Hard — depth 6-8, range 1-1000\n(full distribution shift)"),
    ]

    # Only draw panels for which we actually have data
    available = [
        (ts_key, title)
        for ts_key, title in panels
        if any(
            cond_results.get(ts_key, {}).get("per_depth")
            for cond_results in results.values()
        )
    ]
    if not available:
        print("  Skipping depth curves (no per_depth data available yet)")
        return

    fig, axes = mpl_plt.subplots(1, len(available), figsize=(6 * len(available), 4.5),
                                  sharey=True)
    if len(available) == 1:
        axes = [axes]

    for ax, (ts_key, ts_label) in zip(axes, available):
        for cond, cond_results in results.items():
            raw = cond_results.get(ts_key, {}).get("per_depth", {})
            if not raw:
                continue
            # Normalize keys to int — they may be int (in-memory) or str (from JSON)
            per_depth = {int(k): v for k, v in raw.items()}
            depths = sorted(per_depth.keys())
            accs   = [per_depth[d]["accuracy"] for d in depths]
            counts = [per_depth[d]["count"] for d in depths]
            ax.plot(
                depths, accs,
                marker="o", linewidth=1.8,
                label=CONDITION_LABELS.get(cond, cond),
                color=CONDITION_COLORS.get(cond, None),
            )
            # Annotate sample counts at each depth (first condition only)
            if cond == list(results.keys())[0]:
                for d, a, n in zip(depths, accs, counts):
                    ax.annotate(
                        f"n={n}", xy=(d, a),
                        xytext=(0, 6), textcoords="offset points",
                        fontsize=6, ha="center", color="gray",
                    )

        # Draw the training depth boundary
        ax.axvline(x=5.5, color="gray", linestyle="--", linewidth=1,
                   label="Training boundary (depth 5)")
        ax.set_title(ts_label, fontsize=9)
        ax.set_xlabel("Expression Depth")
        ax.set_ylabel("Exact Match Accuracy")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    path = os.path.join(output_dir, "plot_ood_depth_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    mpl_plt.close(fig)
    print(f"  Saved: {path}")


def plot_ood_taxonomy(
    results: Dict[str, Dict[str, dict]],
    output_dir: str,
    test_set_key: str = "ood_hard",
) -> None:
    """Stacked bar chart of reasoning taxonomy on the hardest OOD set."""
    plt = _require_matplotlib()
    import matplotlib.pyplot as mpl_plt

    taxonomy_keys = ["correct_reasoning", "specious_cot", "unlucky", "failed"]
    taxonomy_labels = ["Correct Reasoning", "Specious CoT", "Unlucky", "Failed"]
    taxonomy_colors = ["#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]

    conditions = list(results.keys())
    x = range(len(conditions))
    bottoms = [0.0] * len(conditions)

    fig, ax = mpl_plt.subplots(figsize=(8, 5))

    for tk, tl, tc in zip(taxonomy_keys, taxonomy_labels, taxonomy_colors):
        values = [
            results[c].get(test_set_key, {}).get("taxonomy_fractions", {}).get(tk, 0)
            for c in conditions
        ]
        ax.bar(x, values, bottom=bottoms, label=tl, color=tc)
        bottoms = [b + v for b, v in zip(bottoms, values)]

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [CONDITION_LABELS.get(c, c) for c in conditions], rotation=15
    )
    ax.set_ylabel("Fraction of Samples")
    ts_label = OOD_SETS.get(test_set_key, {}).get("label", test_set_key)
    ax.set_title(f"Reasoning Taxonomy — {ts_label}")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(output_dir, f"plot_ood_taxonomy_{test_set_key}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    mpl_plt.close(fig)
    print(f"  Saved: {path}")


def plot_ood_error_distribution(
    results: Dict[str, Dict[str, dict]],
    output_dir: str,
    test_set_key: str = "ood_hard",
) -> None:
    """Grouped bar: independent/propagated/invalid errors per condition."""
    plt = _require_matplotlib()
    import matplotlib.pyplot as mpl_plt
    import numpy as np

    conditions = list(results.keys())
    error_types = ["independent", "propagated", "invalid_operands"]
    error_labels = ["Independent", "Propagated", "Invalid Operands"]
    error_colors = ["#d62728", "#ff7f0e", "#9467bd"]

    x = np.arange(len(conditions))
    width = 0.25
    offsets = [-width, 0, width]

    fig, ax = mpl_plt.subplots(figsize=(8, 5))

    for etype, elabel, ecolor, offset in zip(
        error_types, error_labels, error_colors, offsets
    ):
        values = [
            results[c].get(test_set_key, {}).get(
                "error_distribution", {}
            ).get(etype, 0)
            for c in conditions
        ]
        ax.bar(x + offset, values, width, label=elabel, color=ecolor)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [CONDITION_LABELS.get(c, c) for c in conditions], rotation=15
    )
    ax.set_ylabel("Total Error Count")
    ts_label = OOD_SETS.get(test_set_key, {}).get("label", test_set_key)
    ax.set_title(f"Error-Type Distribution — {ts_label}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(output_dir, f"plot_ood_errors_{test_set_key}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    mpl_plt.close(fig)
    print(f"  Saved: {path}")


def print_summary_table(results: Dict[str, Dict[str, dict]]) -> None:
    """Print a readable accuracy table to stdout (bar-chart test sets only)."""
    bar_sets = [ts for ts, cfg in OOD_SETS.items() if cfg.get("bar_chart", True)]
    col_w = 22
    header = f"{'Condition':<20}" + "".join(
        f"{OOD_SETS[ts]['label']:>{col_w}}" for ts in bar_sets
    )
    sep = "=" * len(header)
    print(f"\n{sep}")
    print("OOD ACCURACY SUMMARY")
    print(sep)
    print(header)
    print("-" * len(header))
    for cond, cond_res in results.items():
        row = f"{CONDITION_LABELS.get(cond, cond):<20}"
        for ts in bar_sets:
            acc = cond_res.get(ts, {}).get("exact_match_accuracy", float("nan"))
            if acc != acc:  # nan
                row += f"{'—':>{col_w}}"
            else:
                row += f"{acc:>{col_w}.1%}"
        print(row)
    print(sep)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def run_ood_evaluation(args: argparse.Namespace) -> None:
    experiment_dir = args.experiment_dir
    tokenizer_path = args.tokenizer

    # Discover conditions
    consolidated_path = os.path.join(experiment_dir, "consolidated_results.json")
    if os.path.exists(consolidated_path):
        with open(consolidated_path) as f:
            consolidated = json.load(f)
        conditions = consolidated.get("conditions", [])
    else:
        # Fall back: any subdirectory that has a checkpoint
        conditions = [
            d for d in os.listdir(experiment_dir)
            if os.path.isdir(os.path.join(experiment_dir, d))
            and _find_checkpoint(os.path.join(experiment_dir, d)) is not None
        ]

    if not conditions:
        raise RuntimeError(
            f"No conditions with checkpoints found in {experiment_dir}."
        )

    print(f"Conditions found: {conditions}")
    print(f"Test sets: {list(OOD_SETS.keys())}")
    print(f"Eval samples per test set: {args.eval_samples}")
    print()

    # Skip test sets already completed (resume support)
    ood_results_path = os.path.join(experiment_dir, "ood_results.json")
    if os.path.exists(ood_results_path) and not args.force:
        with open(ood_results_path) as f:
            all_results: Dict[str, Dict[str, dict]] = json.load(f)
        print("Loaded existing ood_results.json (use --force to re-run).")
    else:
        all_results = {}

    for cond in conditions:
        cond_dir = os.path.join(experiment_dir, cond)
        ckpt = _find_checkpoint(cond_dir)
        if ckpt is None:
            print(f"  [{cond}] No checkpoint found — skipping.")
            continue

        print(f"\n{'='*55}")
        print(f"  Condition: {cond}")
        print(f"  Checkpoint: {ckpt}")
        print(f"{'='*55}")

        if cond not in all_results:
            all_results[cond] = {}

        for ts_key, ts_cfg in OOD_SETS.items():
            if ts_key in all_results[cond] and not args.force:
                print(f"    [{ts_key}] already evaluated — skipping.")
                continue

            print(f"    Evaluating {ts_key}: {ts_cfg['label']} ...")
            try:
                report = evaluate_one_set(
                    checkpoint_path=ckpt,
                    tokenizer_path=tokenizer_path,
                    depths=ts_cfg["depths"],
                    num_range=ts_cfg["num_range"],
                    num_samples=args.eval_samples,
                    seed=args.eval_seed,
                )
                result_dict = report.to_dict()
                all_results[cond][ts_key] = result_dict

                acc = report.exact_match_accuracy
                tax = report.taxonomy_fractions
                print(
                    f"      accuracy={acc:.1%}  "
                    f"correct={tax.get('correct_reasoning', 0):.1%}  "
                    f"specious={tax.get('specious_cot', 0):.1%}  "
                    f"unlucky={tax.get('unlucky', 0):.1%}  "
                    f"failed={tax.get('failed', 0):.1%}"
                )

                # Save per-condition per-test-set report
                save_evaluation_report(
                    report,
                    os.path.join(cond_dir, f"eval_{ts_key}.json"),
                    include_samples=False,
                )

            except Exception as exc:
                print(f"      ERROR: {exc}")
                all_results[cond][ts_key] = {"error": str(exc)}

        # Checkpoint after each condition
        with open(ood_results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    # Final save
    with open(ood_results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {ood_results_path}")

    # Update consolidated results
    if os.path.exists(consolidated_path):
        with open(consolidated_path) as f:
            consolidated = json.load(f)
        consolidated["ood_results"] = all_results
        with open(consolidated_path, "w") as f:
            json.dump(consolidated, f, indent=2, default=str)
        print(f"Updated: {consolidated_path}")

    # Print summary table
    print_summary_table(all_results)

    # Generate plots
    print("\nGenerating OOD plots...")
    try:
        plot_ood_accuracy_bars(all_results, experiment_dir)
        plot_ood_depth_curves(all_results, experiment_dir)
        # Taxonomy and error distribution on the hardest test sets
        for ts_key in ("ood_hard", "ood_extreme"):
            if any(
                ts_key in cond_res and "taxonomy_fractions" in cond_res[ts_key]
                for cond_res in all_results.values()
            ):
                plot_ood_taxonomy(all_results, experiment_dir, test_set_key=ts_key)
                plot_ood_error_distribution(all_results, experiment_dir, test_set_key=ts_key)
        print(f"\nAll OOD plots saved to {experiment_dir}")
    except ImportError as e:
        print(f"Plotting skipped: {e}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Re-evaluate GRPO checkpoints on OOD test sets (no retraining).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--experiment-dir", type=str, required=True,
        help="Path to experiment output directory containing condition subdirs.",
    )
    p.add_argument(
        "--tokenizer", type=str, required=True,
        help="Path to trained tokenizer directory.",
    )
    p.add_argument(
        "--eval-samples", type=int, default=500,
        help="Number of evaluation samples per test set.",
    )
    p.add_argument(
        "--eval-seed", type=int, default=456,
        help="Random seed for test-set generation (different from training seed).",
    )
    p.add_argument(
        "--force", action="store_true", default=False,
        help="Re-run evaluations even if ood_results.json already exists.",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    run_ood_evaluation(args)


if __name__ == "__main__":
    main()
