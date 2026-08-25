#!/usr/bin/env python3
"""Experiment orchestrator for process-supervised GRPO.

Runs all four experimental conditions sequentially, then evaluates each
checkpoint with the extended evaluator.  All results are saved to a single
experiment directory for downstream plotting.

Usage::

    python -m arithmetic_llm.run_experiment \
        --tokenizer        path/to/tokenizer \
        --sft-checkpoint   path/to/sft_model.pt \
        --output-dir       experiments/run_01 \
        --data-mode        generated \
        --num-samples      2000 \
        --num-epochs       3 \
        --batch-size       2 \
        --eval-samples     500

All four conditions share the same training data, SFT checkpoint, and
hyperparameters — the only variable is the reward function.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch

from .arithmetic_tokenizer import ArithmeticBPETokenizer
from .evaluator import eval_expression
from .extended_evaluator import ExtendedEvaluator, save_evaluation_report
from .generator import ExpressionGenerator
from .grpo_config import GRPOConfig
from .grpo_trainer import GRPOTrainer
from .reward_functions import build_reward_function
from .reward_scheduler import SchedulerConfig


CONDITIONS = [
    {
        "name": "outcome_only",
        "reward_mode": "outcome_only",
        "description": "Baseline: binary 0/1 reward on final answer",
    },
    {
        "name": "naive_process",
        "reward_mode": "naive_process",
        "description": "Per-step correctness without error-propagation awareness",
    },
    {
        "name": "error_aware",
        "reward_mode": "error_aware",
        "description": "Error-propagation-aware multi-component reward (fixed weights)",
    },
    {
        "name": "scheduled",
        "reward_mode": "scheduled",
        "description": "Error-aware + adaptive reward weight scheduling",
    },
]


def generate_shared_data(
    num_samples: int,
    max_depth: int,
    num_range: Tuple[int, int],
    seed: int = 42,
) -> List[dict]:
    """Generate training pairs deterministically."""
    random.seed(seed)
    gen = ExpressionGenerator(
        max_depth=max_depth,
        num_range=num_range,
        invalid_rate=0.0,
    )
    pairs: List[dict] = []
    attempts = 0
    while len(pairs) < num_samples and attempts < num_samples * 10:
        attempts += 1
        expr = gen.generate()
        result = eval_expression(expr)
        if result["answer"] == "ERROR":
            continue
        pairs.append({
            "prompt": result["problem"] + " <think>",
            "ground_truth": int(result["answer"]),
            "expression": expr,
        })
    return pairs


def _batch_iter(items, batch_size):
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        prompts = [e["prompt"] for e in batch]
        ground_truth = [e["ground_truth"] for e in batch]
        yield prompts, ground_truth


def train_condition(
    condition: dict,
    pairs: List[dict],
    tokenizer: ArithmeticBPETokenizer,
    sft_checkpoint_path: str,
    base_config: GRPOConfig,
    output_dir: str,
    candidate_sub_batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Train a single experimental condition and return metadata."""
    cond_name = condition["name"]
    cond_dir = os.path.join(output_dir, cond_name)
    os.makedirs(cond_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"CONDITION: {cond_name}")
    print(f"  {condition['description']}")
    print(f"  Output: {cond_dir}")
    print(f"{'='*60}\n")

    # Skip training if a completed checkpoint already exists (resume support)
    final_path = os.path.join(cond_dir, "final_model.pt")
    meta_path = os.path.join(cond_dir, "condition_metadata.json")
    if os.path.exists(final_path) and os.path.exists(meta_path):
        print(f"  final_model.pt found — skipping training for '{cond_name}'.")
        with open(meta_path) as f:
            return json.load(f)

    # Build config for this condition
    config = GRPOConfig(
        learning_rate=base_config.learning_rate,
        batch_size=base_config.batch_size,
        num_epochs=base_config.num_epochs,
        warmup_steps=base_config.warmup_steps,
        gradient_clip=base_config.gradient_clip,
        save_every=base_config.save_every,
        eval_every=base_config.eval_every,
        device=base_config.device,
        num_candidates=base_config.num_candidates,
        temperature=base_config.temperature,
        top_k=base_config.top_k,
        top_p=base_config.top_p,
        kl_penalty_coef=base_config.kl_penalty_coef,
        advantage_epsilon=base_config.advantage_epsilon,
        max_gen_length=base_config.max_gen_length,
        gradient_accumulation_steps=base_config.gradient_accumulation_steps,
        log_every=base_config.log_every,
        reward_mode=condition["reward_mode"],
        reward_weights=base_config.reward_weights,
        schedule_strategy=base_config.schedule_strategy,
        schedule_phase1_frac=base_config.schedule_phase1_frac,
        schedule_phase2_frac=base_config.schedule_phase2_frac,
    )
    config.validate()

    total_steps = math.ceil(len(pairs) / config.batch_size) * config.num_epochs

    # Build reward function
    reward_fn = None
    if condition["reward_mode"] != "outcome_only":
        scheduler_config = None
        if condition["reward_mode"] == "scheduled":
            scheduler_config = SchedulerConfig(
                strategy=config.schedule_strategy,
                total_steps=total_steps,
                phase1_frac=config.schedule_phase1_frac,
                phase2_frac=config.schedule_phase2_frac,
            )
        reward_fn = build_reward_function(
            mode=condition["reward_mode"],
            weights=config.reward_weights,
            scheduler_config=scheduler_config,
        )

    trainer = GRPOTrainer(
        config=config,
        sft_checkpoint_path=sft_checkpoint_path,
        tokenizer=tokenizer,
        candidate_sub_batch_size=candidate_sub_batch_size,
        reward_fn=reward_fn,
    )
    trainer.reset_optimizer_and_scheduler(total_steps=total_steps)

    train_dataloader = list(_batch_iter(pairs, config.batch_size))

    start_time = time.time()
    train_result = trainer.train(train_dataloader, output_dir=cond_dir)
    elapsed = time.time() - start_time

    metadata = {
        "condition": cond_name,
        "description": condition["description"],
        "reward_mode": condition["reward_mode"],
        "training_time_seconds": elapsed,
        "output_dir": cond_dir,
        "config": config.to_dict(),
        **train_result,
    }

    meta_path = os.path.join(cond_dir, "condition_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return metadata


def evaluate_condition(
    cond_dir: str,
    cond_name: str,
    tokenizer_path: str,
    eval_samples: int = 500,
    eval_depths: Optional[List[int]] = None,
    eval_num_range: Tuple[int, int] = (1, 20),
    eval_seed: int = 123,
    base_checkpoint_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a trained condition checkpoint with the extended evaluator."""
    from .evaluator import ModelEvaluator

    # Find the final model checkpoint
    final_path = os.path.join(cond_dir, "final_model.pt")
    if not os.path.exists(final_path):
        # Try to find latest checkpoint
        import glob
        checkpoints = sorted(glob.glob(os.path.join(cond_dir, "checkpoint_step_*.pt")))
        if checkpoints:
            final_path = checkpoints[-1]
        else:
            print(f"  No checkpoint found in {cond_dir}, skipping evaluation")
            return {}

    print(f"\n  Evaluating {cond_name} from {final_path}")

    evaluator_model = ModelEvaluator(
        model_path=final_path,
        tokenizer_path=tokenizer_path,
        base_checkpoint_path=base_checkpoint_path,
    )

    def generate_fn(prompt: str) -> str:
        return evaluator_model._generate_solution(prompt, max_length=512)

    ext_eval = ExtendedEvaluator(generate_fn)
    report = ext_eval.evaluate(
        num_samples=eval_samples,
        max_depth=max(eval_depths) if eval_depths else 5,
        num_range=eval_num_range,
        depths=eval_depths,
        seed=eval_seed,
    )

    # Save report
    report_path = os.path.join(cond_dir, "extended_eval_report.json")
    save_evaluation_report(report, report_path, include_samples=True)
    print(f"  Accuracy: {report.exact_match_accuracy:.2%}")
    print(f"  Taxonomy: {report.taxonomy_fractions}")

    return report.to_dict()


def run_full_experiment(args: argparse.Namespace) -> None:
    """Run all conditions, evaluate, and save consolidated results."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save experiment config
    experiment_config = vars(args)
    with open(os.path.join(output_dir, "experiment_config.json"), "w") as f:
        json.dump(experiment_config, f, indent=2, default=str)

    # Load tokenizer
    tokenizer = ArithmeticBPETokenizer()
    tokenizer.load(args.tokenizer)

    # Generate shared training data
    print("Generating shared training data...")
    pairs = generate_shared_data(
        num_samples=args.num_samples,
        max_depth=args.max_depth,
        num_range=(args.num_range_min, args.num_range_max),
        seed=args.seed,
    )
    print(f"  Generated {len(pairs)} training pairs")

    # Save training data for reproducibility
    data_path = os.path.join(output_dir, "training_data.json")
    with open(data_path, "w") as f:
        json.dump(pairs, f, indent=2)

    # Build base config
    base_config = GRPOConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        gradient_clip=args.gradient_clip,
        save_every=args.save_every,
        eval_every=args.eval_every,
        num_candidates=args.num_candidates,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        kl_penalty_coef=args.kl_penalty_coef,
        max_gen_length=args.max_gen_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_every=args.log_every,
        reward_weights=tuple(args.reward_weights),
        schedule_strategy=args.schedule_strategy,
        schedule_phase1_frac=args.schedule_phase1_frac,
        schedule_phase2_frac=args.schedule_phase2_frac,
    )

    # Determine which conditions to run
    if args.conditions:
        conditions = [c for c in CONDITIONS if c["name"] in args.conditions]
    else:
        conditions = CONDITIONS

    # Train all conditions
    all_metadata = {}
    for condition in conditions:
        meta = train_condition(
            condition=condition,
            pairs=pairs,
            tokenizer=tokenizer,
            sft_checkpoint_path=args.sft_checkpoint,
            base_config=base_config,
            output_dir=output_dir,
            candidate_sub_batch_size=args.candidate_sub_batch_size,
        )
        all_metadata[condition["name"]] = meta

    # Evaluate all conditions
    print(f"\n{'='*60}")
    print("EVALUATION PHASE")
    print(f"{'='*60}")

    eval_depths = list(range(1, args.max_depth + 1)) if args.eval_per_depth else None
    all_eval_results = {}

    for condition in conditions:
        cond_name = condition["name"]
        cond_dir = os.path.join(output_dir, cond_name)

        eval_result = evaluate_condition(
            cond_dir=cond_dir,
            cond_name=cond_name,
            tokenizer_path=args.tokenizer,
            eval_samples=args.eval_samples,
            eval_depths=eval_depths,
            eval_num_range=(args.num_range_min, args.num_range_max),
            eval_seed=args.eval_seed,
        )
        all_eval_results[cond_name] = eval_result

    # Save consolidated results
    consolidated = {
        "experiment_timestamp": datetime.now().isoformat(),
        "conditions": list(all_metadata.keys()),
        "training_metadata": all_metadata,
        "evaluation_results": all_eval_results,
    }
    consolidated_path = os.path.join(output_dir, "consolidated_results.json")
    with open(consolidated_path, "w") as f:
        json.dump(consolidated, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"Consolidated results: {consolidated_path}")
    print(f"Run `python -m arithmetic_llm.plot_results {output_dir}` to generate plots.")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run full process-supervision GRPO experiment"
    )

    # Required paths
    p.add_argument("--tokenizer", type=str, required=True)
    p.add_argument("--sft-checkpoint", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)

    # Data
    p.add_argument("--data-mode", type=str, default="generated",
                    choices=["instruction", "generated"])
    p.add_argument("--instruction-corpus", type=str, default=None)
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--num-range-min", type=int, default=1)
    p.add_argument("--num-range-max", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)

    # Training
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--gradient-clip", type=float, default=1.0)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--num-candidates", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--kl-penalty-coef", type=float, default=0.05)
    p.add_argument("--max-gen-length", type=int, default=400)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--candidate-sub-batch-size", type=int, default=None)

    # Reward
    p.add_argument("--reward-weights", type=float, nargs=4,
                    default=[0.10, 0.40, 0.20, 0.30],
                    metavar=("FMT", "PROC", "CONS", "OUT"))
    p.add_argument("--schedule-strategy", type=str, default="linear",
                    choices=["linear", "cosine", "threshold", "fixed"])
    p.add_argument("--schedule-phase1-frac", type=float, default=0.25)
    p.add_argument("--schedule-phase2-frac", type=float, default=0.50)

    # Conditions to run (default: all)
    p.add_argument("--conditions", type=str, nargs="*", default=None,
                    choices=["outcome_only", "naive_process", "error_aware", "scheduled"],
                    help="Subset of conditions to run (default: all)")

    # Evaluation
    p.add_argument("--eval-samples", type=int, default=500)
    p.add_argument("--eval-seed", type=int, default=123)
    p.add_argument("--eval-per-depth", action="store_true", default=True,
                    help="Evaluate per expression depth")

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    run_full_experiment(args)


if __name__ == "__main__":
    main()
