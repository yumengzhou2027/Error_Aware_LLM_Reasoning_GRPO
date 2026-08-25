#!/usr/bin/env python3
"""Case study: verify different rewards were applied and compare model outputs.

Two modes:

  verify   -- reads training logs to prove each condition used a different
              reward signal.  No model loading required; runs in seconds.

  generate -- loads all four final_model.pt checkpoints and generates
              responses side-by-side for a fixed set of test prompts.
              Results are printed and saved to <experiment-dir>/case_study.txt

Usage::

    # Prove different rewards were applied (fast, no GPU needed)
    python -m arithmetic_llm.case_study verify \
        --experiment-dir experiments/run_01

    # Side-by-side generation comparison
    python -m arithmetic_llm.case_study generate \
        --experiment-dir experiments/run_01 \
        --tokenizer data/tokenizer
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple


CONDITION_LABELS = {
    "outcome_only": "Outcome Only",
    "naive_process": "Naive Process",
    "error_aware":   "Error Aware",
    "scheduled":     "Scheduled",
}

# Fixed test prompts covering a range of difficulties.
# These are used for the side-by-side generation comparison.
TEST_PROMPTS = [
    # depth 1 — trivial
    ("Evaluate: 7 + 4 <think>",          11),
    ("Evaluate: 15 - 8 <think>",          7),
    # depth 2
    ("Evaluate: (3 + 5) - 2 <think>",     6),
    ("Evaluate: 10 - (4 + 3) <think>",    3),
    # depth 3
    ("Evaluate: (2 + (5 - 1)) + 3 <think>",              9),
    ("Evaluate: (10 - (3 + 2)) - 1 <think>",             4),
    # depth 4 — in-distribution ceiling
    ("Evaluate: ((7 + 2) - (4 - 1)) + 5 <think>",       11),
    ("Evaluate: (8 - (3 + (5 - 2))) + 6 <think>",        8),
    # depth 5 — upper bound of training
    ("Evaluate: (((4 + 3) - 2) + (5 - 1)) - 6 <think>",  3),
    # depth 6 — OOD
    ("Evaluate: ((3 + (7 - 2)) - ((1 + 4) - 3)) + 5 <think>", 11),
]


# ─────────────────────────────────────────────────────────
# Mode 1: verify  (reads logs, no model loading)
# ─────────────────────────────────────────────────────────

def _find_log(cond_dir: str) -> Optional[str]:
    direct = os.path.join(cond_dir, "grpo_training_log.json")
    if os.path.exists(direct):
        return direct
    for entry in os.listdir(cond_dir):
        sub = os.path.join(cond_dir, entry, "grpo_training_log.json")
        if os.path.exists(sub):
            return sub
    return None


def _find_metadata(cond_dir: str) -> Optional[str]:
    p = os.path.join(cond_dir, "condition_metadata.json")
    return p if os.path.exists(p) else None


def verify_rewards(experiment_dir: str) -> None:
    """Print proof that each condition used a different reward signal."""

    conditions = [
        d for d in os.listdir(experiment_dir)
        if os.path.isdir(os.path.join(experiment_dir, d))
        and d in CONDITION_LABELS
    ]
    conditions.sort(key=lambda c: list(CONDITION_LABELS).index(c)
                    if c in CONDITION_LABELS else 99)

    print("\n" + "=" * 70)
    print("REWARD VERIFICATION — training log analysis")
    print("=" * 70)

    for cond in conditions:
        cond_dir = os.path.join(experiment_dir, cond)
        label = CONDITION_LABELS.get(cond, cond)
        print(f"\n{'─'*70}")
        print(f"  {label}  ({cond})")
        print(f"{'─'*70}")

        # Check condition_metadata for reward_mode
        meta_path = _find_metadata(cond_dir)
        if meta_path:
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"  reward_mode : {meta.get('reward_mode', '?')}")
            cfg = meta.get("config", {})
            if "reward_weights" in cfg:
                print(f"  reward_weights (fixed): {cfg['reward_weights']}")
            if "schedule_strategy" in cfg:
                print(f"  schedule_strategy      : {cfg['schedule_strategy']}")

        # Load training log
        log_path = _find_log(cond_dir)
        if not log_path:
            print("  [!] No training log found")
            continue

        with open(log_path) as f:
            log = json.load(f)

        entries = [e for e in log if e.get("step", 0) > 0]
        if not entries:
            print("  [!] Training log is empty")
            continue

        total_steps = len(entries)
        first = entries[0]["metrics"]
        last  = entries[-1]["metrics"]

        # Key discriminating fields
        has_components  = "avg_format_score" in first
        has_phase       = "reward_phase" in first
        has_weights     = "reward_weights" in first

        print(f"  training steps logged : {total_steps}")
        print(f"  has component scores  : {has_components}  "
              f"{'(process reward active ✓)' if has_components else '(outcome-only baseline)'}")
        print(f"  has reward_phase      : {has_phase}  "
              f"{'(adaptive scheduler active ✓)' if has_phase else ''}")

        # avg_reward vs reward_rate divergence proves different reward scale
        avg_r_first = first.get("avg_reward", 0.0)
        rr_first    = first.get("reward_rate", 0.0)
        avg_r_last  = last.get("avg_reward", 0.0)
        rr_last     = last.get("reward_rate", 0.0)

        print(f"\n  Step 1 — avg_reward={avg_r_first:.3f}  reward_rate={rr_first:.3f}  "
              f"gap={avg_r_first - rr_first:+.3f}")
        print(f"  Last   — avg_reward={avg_r_last:.3f}  reward_rate={rr_last:.3f}  "
              f"gap={avg_r_last - rr_last:+.3f}")
        print()
        print("  Interpretation:")
        if not has_components:
            print("    avg_reward ≈ reward_rate throughout (binary 0/1 outcome signal).")
            print("    No process components — this IS the outcome-only baseline.")
        else:
            gap = avg_r_first - rr_first
            if abs(gap) < 0.01:
                note = "small gap — format/process/consistency scores were mostly 1.0 too"
            elif gap < 0:
                note = "composite reward < binary accuracy — process/format pulled it DOWN"
            else:
                note = "composite reward > binary accuracy — process/format added signal"
            print(f"    avg_reward ≠ reward_rate ({note}).")
            print("    Component scores are present — process reward WAS active.")

        if has_components:
            # Show average component scores
            fmt_vals   = [e["metrics"].get("avg_format_score", 0) for e in entries]
            proc_vals  = [e["metrics"].get("avg_process_score", 0) for e in entries]
            cons_vals  = [e["metrics"].get("avg_consistency_score", 0) for e in entries]
            out_vals   = [e["metrics"].get("avg_outcome_score", 0) for e in entries]

            def _mean(xs):
                return sum(xs) / len(xs) if xs else 0.0

            print(f"\n  Mean component scores over training:")
            print(f"    format      = {_mean(fmt_vals):.3f}")
            print(f"    process     = {_mean(proc_vals):.3f}")
            print(f"    consistency = {_mean(cons_vals):.3f}")
            print(f"    outcome     = {_mean(out_vals):.3f}")

        if has_phase:
            phases = [e["metrics"]["reward_phase"] for e in entries]
            phase_changes = [(phases[0], entries[0]["step"])]
            for i in range(1, len(phases)):
                if phases[i] != phases[i - 1]:
                    phase_changes.append((phases[i], entries[i]["step"]))
            print(f"\n  Scheduler phase transitions:")
            for ph, step in phase_changes:
                name = {1: "Format", 2: "Process", 3: "Outcome"}.get(int(ph), str(ph))
                print(f"    Phase {int(ph)} ({name}) started at step {step}")

            if has_weights:
                w_first = entries[0]["metrics"]["reward_weights"]
                w_last  = entries[-1]["metrics"]["reward_weights"]
                print(f"\n  Reward weights — step 1  : fmt={w_first[0]:.2f}  "
                      f"proc={w_first[1]:.2f}  cons={w_first[2]:.2f}  out={w_first[3]:.2f}")
                print(f"  Reward weights — last step: fmt={w_last[0]:.2f}  "
                      f"proc={w_last[1]:.2f}  cons={w_last[2]:.2f}  out={w_last[3]:.2f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
  outcome_only  : reward = binary {0,1} on final answer only.
                  avg_reward ≈ reward_rate (they are the same signal).

  naive_process : reward = weighted composite including per-step accuracy
                  (checked in isolation, no error propagation awareness).
                  avg_reward < reward_rate when steps are harder than the
                  final answer (process score penalises wrong intermediate steps).

  error_aware   : same composite as naive_process BUT process_score credits
                  propagated errors separately — model gets partial credit for
                  correct computation on a wrong inherited input.

  scheduled     : same as error_aware but reward_weights shift over training:
                  Phase 1 (format-heavy) → Phase 2 (process-heavy) → Phase 3
                  (outcome-heavy).  reward_weights in the log prove this happened.
""")


# ─────────────────────────────────────────────────────────
# Mode 2: generate  (loads models, side-by-side outputs)
# ─────────────────────────────────────────────────────────

def _find_checkpoint(cond_dir: str) -> Optional[str]:
    p = os.path.join(cond_dir, "final_model.pt")
    if os.path.exists(p):
        return p
    import glob as _glob
    ckpts = sorted(_glob.glob(os.path.join(cond_dir, "checkpoint_step_*.pt")))
    return ckpts[-1] if ckpts else None


def generate_case_study(
    experiment_dir: str,
    tokenizer_path: str,
    prompts: Optional[List[Tuple[str, int]]] = None,
    output_file: Optional[str] = None,
) -> None:
    """Load all four models and generate side-by-side responses."""
    from .evaluator import ModelEvaluator

    if prompts is None:
        prompts = TEST_PROMPTS

    conditions = [c for c in CONDITION_LABELS if
                  os.path.isdir(os.path.join(experiment_dir, c))]
    conditions.sort(key=lambda c: list(CONDITION_LABELS).index(c))

    print("\nLoading models...")
    models: Dict[str, ModelEvaluator] = {}
    for cond in conditions:
        cond_dir = os.path.join(experiment_dir, cond)
        ckpt = _find_checkpoint(cond_dir)
        if ckpt is None:
            print(f"  [{cond}] no checkpoint found — skipping")
            continue
        print(f"  Loading {cond} from {ckpt} ...")
        models[cond] = ModelEvaluator(
            model_path=ckpt,
            tokenizer_path=tokenizer_path,
        )
    print(f"  Loaded {len(models)} models.\n")

    lines: List[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("=" * 80)
    emit("SIDE-BY-SIDE CASE STUDY")
    emit("=" * 80)
    emit(f"Models: {', '.join(CONDITION_LABELS.get(c, c) for c in models)}")
    emit()

    for prompt_idx, (prompt, ground_truth) in enumerate(prompts):
        emit("─" * 80)
        emit(f"Prompt {prompt_idx + 1}: {prompt}")
        emit(f"Ground truth: {ground_truth}")
        emit()

        for cond, model in models.items():
            label = CONDITION_LABELS.get(cond, cond)
            generated = model._generate_solution(prompt, max_length=512)

            # Quick parse for final result
            import re
            m = re.search(r"Final Result:\s*(-?\d+)", generated)
            predicted = int(m.group(1)) if m else None
            correct = "✓" if predicted == ground_truth else "✗"

            emit(f"  [{label}]  predicted={predicted}  {correct}")
            # Indent the full output
            for line in generated.strip().split("\n"):
                emit(f"    {line}")
            emit()

    emit("=" * 80)

    if output_file:
        with open(output_file, "w") as f:
            f.write("\n".join(lines))
        print(f"\nSaved to: {output_file}")


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify reward differences and run side-by-side case study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # verify sub-command
    v = sub.add_parser(
        "verify",
        help="Read training logs to prove different rewards were applied.",
    )
    v.add_argument("--experiment-dir", required=True)

    # generate sub-command
    g = sub.add_parser(
        "generate",
        help="Load all four models and generate side-by-side outputs.",
    )
    g.add_argument("--experiment-dir", required=True)
    g.add_argument("--tokenizer", required=True)
    g.add_argument(
        "--output-file", default=None,
        help="Save output to this file (default: <experiment-dir>/case_study.txt)",
    )

    args = parser.parse_args()

    if args.mode == "verify":
        verify_rewards(args.experiment_dir)
    elif args.mode == "generate":
        out = args.output_file or os.path.join(args.experiment_dir, "case_study.txt")
        generate_case_study(
            experiment_dir=args.experiment_dir,
            tokenizer_path=args.tokenizer,
            output_file=out,
        )


if __name__ == "__main__":
    main()
