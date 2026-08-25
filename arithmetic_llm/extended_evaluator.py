"""Extended evaluation module for process-supervised GRPO experiments.

Adds three capabilities beyond the base :class:`ModelEvaluator`:

1. **Reasoning quality taxonomy** — classifies each output as:
   - correct_reasoning    : right answer + all steps correct
   - specious_cot         : right answer + at least one independent error
   - unlucky              : wrong answer + all steps correct
   - failed               : wrong answer + at least one independent error

2. **Error-type distribution** — independent vs. propagated vs. invalid

3. **Per-depth breakdown** — accuracy and taxonomy at each expression depth
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .evaluator import eval_expression
from .expression_state_tracker import ExpressionStateTracker, SolutionVerification
from .generator import ExpressionGenerator
from .step_parser import ParsedSolution, parse_solution


TAXONOMY_LABELS = (
    "correct_reasoning",
    "specious_cot",
    "unlucky",
    "failed",
)


@dataclass
class SampleResult:
    """Detailed evaluation result for a single sample."""
    expression: str
    depth: int
    ground_truth: int
    generated_text: str
    predicted_result: Optional[int]
    outcome_correct: bool
    taxonomy: str  # one of TAXONOMY_LABELS
    independent_errors: int = 0
    propagated_errors: int = 0
    invalid_operand_errors: int = 0
    total_steps_generated: int = 0
    total_steps_expected: int = 0
    computational_correctness: float = 0.0


@dataclass
class EvaluationReport:
    """Aggregate evaluation results across all samples."""
    samples: List[SampleResult] = field(default_factory=list)

    total: int = 0
    exact_match_accuracy: float = 0.0
    parse_success_rate: float = 0.0

    taxonomy_counts: Dict[str, int] = field(default_factory=lambda: {
        k: 0 for k in TAXONOMY_LABELS
    })
    taxonomy_fractions: Dict[str, float] = field(default_factory=lambda: {
        k: 0.0 for k in TAXONOMY_LABELS
    })

    total_independent_errors: int = 0
    total_propagated_errors: int = 0
    total_invalid_operand_errors: int = 0

    per_depth: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "exact_match_accuracy": self.exact_match_accuracy,
            "parse_success_rate": self.parse_success_rate,
            "taxonomy_counts": self.taxonomy_counts,
            "taxonomy_fractions": self.taxonomy_fractions,
            "error_distribution": {
                "independent": self.total_independent_errors,
                "propagated": self.total_propagated_errors,
                "invalid_operands": self.total_invalid_operand_errors,
            },
            "per_depth": self.per_depth,
        }


class ExtendedEvaluator:
    """Evaluate a generation function with reasoning quality analysis."""

    def __init__(self, generate_fn):
        self._generate = generate_fn

    def evaluate(
        self,
        num_samples: int = 500,
        max_depth: int = 5,
        num_range: Tuple[int, int] = (1, 20),
        depths: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> EvaluationReport:
        """Run evaluation and produce a detailed report."""
        import random
        if seed is not None:
            random.seed(seed)

        # Generate test set
        test_cases = self._generate_test_set(
            num_samples, max_depth, num_range, depths
        )

        report = EvaluationReport()
        report.total = len(test_cases)
        parseable_count = 0

        depth_buckets: Dict[int, List[SampleResult]] = defaultdict(list)

        for expr, gt, depth in test_cases:
            prompt = f"Evaluate: {expr} <think>"
            generated_text = self._generate(prompt)

            # Parse + verify
            parsed = parse_solution(generated_text)
            tracker = ExpressionStateTracker(expr)
            verification = tracker.verify_solution(parsed, gt)

            predicted = parsed.final_result
            outcome_correct = (predicted == gt)

            if predicted is not None:
                parseable_count += 1

            # Classify into taxonomy
            has_independent_error = verification.independent_error_count > 0
            if outcome_correct and not has_independent_error:
                taxonomy = "correct_reasoning"
            elif outcome_correct and has_independent_error:
                taxonomy = "specious_cot"
            elif not outcome_correct and not has_independent_error and verification.total_steps_generated > 0:
                taxonomy = "unlucky"
            else:
                taxonomy = "failed"

            sr = SampleResult(
                expression=expr,
                depth=depth,
                ground_truth=gt,
                generated_text=generated_text,
                predicted_result=predicted,
                outcome_correct=outcome_correct,
                taxonomy=taxonomy,
                independent_errors=verification.independent_error_count,
                propagated_errors=verification.propagated_error_count,
                invalid_operand_errors=verification.invalid_operand_count,
                total_steps_generated=verification.total_steps_generated,
                total_steps_expected=verification.total_steps_expected,
                computational_correctness=verification.computational_correctness,
            )
            report.samples.append(sr)
            depth_buckets[depth].append(sr)

            # Accumulate
            report.taxonomy_counts[taxonomy] += 1
            report.total_independent_errors += verification.independent_error_count
            report.total_propagated_errors += verification.propagated_error_count
            report.total_invalid_operand_errors += verification.invalid_operand_count

        # Finalize summary
        n = max(report.total, 1)
        report.exact_match_accuracy = (
            sum(1 for s in report.samples if s.outcome_correct) / n
        )
        report.parse_success_rate = parseable_count / n

        for k in TAXONOMY_LABELS:
            report.taxonomy_fractions[k] = report.taxonomy_counts[k] / n

        # Per-depth breakdown
        for depth, bucket in sorted(depth_buckets.items()):
            bd = len(bucket)
            correct = sum(1 for s in bucket if s.outcome_correct)
            tax = {k: sum(1 for s in bucket if s.taxonomy == k) for k in TAXONOMY_LABELS}
            report.per_depth[depth] = {
                "count": bd,
                "accuracy": correct / max(bd, 1),
                "taxonomy_counts": tax,
                "taxonomy_fractions": {k: v / max(bd, 1) for k, v in tax.items()},
                "avg_computational_correctness": (
                    sum(s.computational_correctness for s in bucket) / max(bd, 1)
                ),
            }

        return report

    @staticmethod
    def _generate_test_set(
        num_samples: int,
        max_depth: int,
        num_range: Tuple[int, int],
        depths: Optional[List[int]],
    ) -> List[Tuple[str, int, int]]:
        """Return list of (expression, ground_truth, depth)."""
        cases: List[Tuple[str, int, int]] = []

        if depths is not None:
            per_depth = max(1, num_samples // len(depths))
            for d in depths:
                gen = ExpressionGenerator(
                    max_depth=d,
                    num_range=num_range,
                    invalid_rate=0.0,
                )
                count = 0
                attempts = 0
                while count < per_depth and attempts < per_depth * 10:
                    attempts += 1
                    expr = gen.generate()
                    result = eval_expression(expr)
                    if result["answer"] == "ERROR":
                        continue
                    cases.append((expr, int(result["answer"]), d))
                    count += 1
        else:
            gen = ExpressionGenerator(
                max_depth=max_depth,
                num_range=num_range,
                invalid_rate=0.0,
            )
            attempts = 0
            while len(cases) < num_samples and attempts < num_samples * 10:
                attempts += 1
                expr = gen.generate()
                result = eval_expression(expr)
                if result["answer"] == "ERROR":
                    continue
                # Estimate depth from step count
                depth = len(result["solution"].split("Step")) - 1
                cases.append((expr, int(result["answer"]), max(depth, 1)))

        return cases


def save_evaluation_report(
    report: EvaluationReport,
    output_path: str,
    include_samples: bool = False,
) -> None:
    """Save an evaluation report to JSON."""
    data = report.to_dict()
    if include_samples:
        data["samples"] = [
            {
                "expression": s.expression,
                "depth": s.depth,
                "ground_truth": s.ground_truth,
                "predicted_result": s.predicted_result,
                "outcome_correct": s.outcome_correct,
                "taxonomy": s.taxonomy,
                "independent_errors": s.independent_errors,
                "propagated_errors": s.propagated_errors,
                "computational_correctness": s.computational_correctness,
                "generated_text": s.generated_text,
            }
            for s in report.samples
        ]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
