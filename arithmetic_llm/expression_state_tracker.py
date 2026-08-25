"""Error-propagation-aware verification of model-generated arithmetic solutions.

Given an original expression and the model's step-by-step output, this module
classifies each reasoning step as:

- ``correct``            – arithmetic is right and operands are valid
- ``independent_error``  – the model computed A op B incorrectly
- ``propagated_error``   – the arithmetic A op B = C is correct, but at least
                           one operand was itself wrong (inherited from a prior
                           incorrect step)
- ``invalid_operands``   – the operands don't correspond to any sub-expression
                           in the model's own running expression state
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .evaluator import eval_expression
from .step_parser import ParsedSolution, ParsedStep, parse_solution


@dataclass
class StepVerification:
    """Verification result for a single reasoning step."""
    step_number: int
    is_arithmetic_correct: bool     # does A op B actually equal C?
    error_type: str                 # 'correct' | 'independent_error' | 'propagated_error' | 'invalid_operands'
    claimed: str = ""               # e.g. "7 + 3 = 10"
    expected_result: Optional[int] = None   # what A op B actually equals


@dataclass
class SolutionVerification:
    """Aggregate verification of a full model solution."""
    step_verifications: List[StepVerification] = field(default_factory=list)

    # Summary metrics
    computational_correctness: float = 0.0   # fraction of steps where arithmetic is correct
    outcome_correct: bool = False            # final answer matches ground truth
    independent_error_count: int = 0
    propagated_error_count: int = 0
    correct_step_count: int = 0
    invalid_operand_count: int = 0
    total_steps_generated: int = 0
    total_steps_expected: int = 0            # from canonical evaluation


class ExpressionStateTracker:
    """Track tainted values to classify steps as correct, independent error, propagated error, or invalid."""

    def __init__(self, original_expression: str):
        self.original_expression = original_expression

        # Generate canonical solution for reference (called once; stored for reuse)
        canonical = eval_expression(original_expression)
        self._canonical_answer = canonical["answer"]
        self._canonical_solution: str = canonical.get("solution", "")

        # Leaf numbers that appear in the original expression — these are always
        # valid operands regardless of what the model has produced so far.
        import re as _re
        self._original_numbers: Set[int] = {
            int(m) for m in _re.findall(r"-?\d+", original_expression)
        }

        # Set of intermediate values from the canonical evaluation.
        # We use this to detect when an operand the model uses is "wrong"
        # (i.e. never appears in the correct evaluation trace).
        self._canonical_intermediates: Set[int] = set()
        if canonical["answer"] != "ERROR":
            self._parse_canonical_intermediates(self._canonical_solution)

    def verify_solution(
        self,
        parsed: ParsedSolution,
        ground_truth: int,
    ) -> SolutionVerification:
        """Verify every step of a parsed model solution against the ground truth."""
        result = SolutionVerification()
        result.outcome_correct = (parsed.final_result == ground_truth)
        result.total_steps_generated = len(parsed.steps)
        result.total_steps_expected = self._count_canonical_steps()

        # Track which intermediate *results* the model has produced that
        # came from an incorrect step.  These are "tainted".
        tainted_values: Set[int] = set()
        # Track all results the model has produced (for operand matching).
        model_produced: Set[int] = set()

        for ps in parsed.steps:
            sv = self._verify_step(ps, tainted_values, model_produced)
            result.step_verifications.append(sv)

            # Update tainted / produced sets
            model_produced.add(ps.claimed_result)
            if sv.error_type == "independent_error":
                # The result is tainted because the computation was wrong
                tainted_values.add(ps.claimed_result)
                result.independent_error_count += 1
            elif sv.error_type == "propagated_error":
                # The result is also tainted (built on a tainted input)
                tainted_values.add(ps.claimed_result)
                result.propagated_error_count += 1
            elif sv.error_type == "invalid_operands":
                tainted_values.add(ps.claimed_result)
                result.invalid_operand_count += 1
            else:
                result.correct_step_count += 1

        total = result.total_steps_generated
        if total > 0:
            result.computational_correctness = (
                sum(1 for sv in result.step_verifications if sv.is_arithmetic_correct)
                / total
            )
        else:
            result.computational_correctness = 0.0

        return result

    def _verify_step(
        self,
        step: ParsedStep,
        tainted_values: Set[int],
        model_produced: Set[int],
    ) -> StepVerification:
        """Classify a single step."""
        a, op, b, c = (
            step.left_operand,
            step.operator,
            step.right_operand,
            step.claimed_result,
        )
        expected = a + b if op == "+" else a - b
        arithmetic_ok = (expected == c)

        claimed_str = f"{a} {op} {b} = {c}"

        if arithmetic_ok:
            # The computation itself is correct.  But are the operands valid?
            # An operand is "tainted" if it came from a prior wrong step.
            uses_tainted = (a in tainted_values) or (b in tainted_values)
            if uses_tainted:
                return StepVerification(
                    step_number=step.step_number,
                    is_arithmetic_correct=True,
                    error_type="propagated_error",
                    claimed=claimed_str,
                    expected_result=expected,
                )
            return StepVerification(
                step_number=step.step_number,
                is_arithmetic_correct=True,
                error_type="correct",
                claimed=claimed_str,
                expected_result=expected,
            )
        else:
            # The arithmetic itself is wrong.  Distinguish whether the operands
            # were valid (came from the original expression or a prior model
            # step) or were hallucinated entirely.
            valid_operands = self._original_numbers | model_produced
            if (a in valid_operands) and (b in valid_operands):
                error_type = "independent_error"
            else:
                error_type = "invalid_operands"
            return StepVerification(
                step_number=step.step_number,
                is_arithmetic_correct=False,
                error_type=error_type,
                claimed=claimed_str,
                expected_result=expected,
            )

    def _parse_canonical_intermediates(self, solution_text: str) -> None:
        """Extract all intermediate numeric results from the canonical solution."""
        import re
        for m in re.finditer(r"Step\s+\d+:\s*(-?\d+)\s*[+\-]\s*(-?\d+)\s*=\s*(-?\d+)", solution_text):
            self._canonical_intermediates.add(int(m.group(1)))
            self._canonical_intermediates.add(int(m.group(2)))
            self._canonical_intermediates.add(int(m.group(3)))

    def _count_canonical_steps(self) -> int:
        """Count steps in the canonical solution (uses cached solution text)."""
        if self._canonical_answer == "ERROR":
            return 0
        import re
        return len(re.findall(r"Step\s+\d+:", self._canonical_solution))
