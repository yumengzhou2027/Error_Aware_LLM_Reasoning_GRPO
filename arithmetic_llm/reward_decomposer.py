"""Multi-component reward decomposition for arithmetic GRPO.

Decomposes the reward signal into four orthogonal components:
- **format**      : structural compliance (think tags, step markers, final result line)
- **process**     : fraction of steps with correct arithmetic (error-propagation-aware)
- **consistency** : agreement between claimed step results and "Expression now:" lines
- **outcome**     : binary correctness of the final answer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .expression_state_tracker import (
    ExpressionStateTracker,
    SolutionVerification,
    StepVerification,
)
from .step_parser import ParsedSolution, parse_solution


@dataclass
class RewardVector:
    """Structured reward with per-component scores in [0, 1]."""
    format_score: float = 0.0
    process_score: float = 0.0
    consistency_score: float = 0.0
    outcome_score: float = 0.0

    # Detailed diagnostics (not used in scalar reward, useful for logging)
    verification: Optional[SolutionVerification] = field(default=None, repr=False)
    parsed: Optional[ParsedSolution] = field(default=None, repr=False)

    def to_scalar(self, weights: Tuple[float, float, float, float]) -> float:
        """Weighted combination of the four components into a scalar reward."""
        w_fmt, w_proc, w_cons, w_out = weights
        total_w = w_fmt + w_proc + w_cons + w_out
        if total_w == 0:
            return 0.0
        return (
            w_fmt * self.format_score
            + w_proc * self.process_score
            + w_cons * self.consistency_score
            + w_out * self.outcome_score
        ) / total_w

    def to_dict(self) -> dict:
        """Serialize the four scores for JSON logging."""
        return {
            "format": self.format_score,
            "process": self.process_score,
            "consistency": self.consistency_score,
            "outcome": self.outcome_score,
        }


class RewardDecomposer:
    """Compute :class:`RewardVector` from a generated solution.

    This is the central class that wires together parsing, verification,
    and scoring into a single ``compute()`` call.
    """

    def compute(
        self,
        generated_text: str,
        original_expression: str,
        ground_truth: int,
    ) -> RewardVector:
        """Produce a multi-component reward for one candidate solution."""
        parsed = parse_solution(generated_text)
        tracker = ExpressionStateTracker(original_expression)
        verification = tracker.verify_solution(parsed, ground_truth)

        rv = RewardVector(
            format_score=self._format_score(parsed),
            process_score=self._process_score(verification),
            consistency_score=self._consistency_score(parsed),
            outcome_score=1.0 if verification.outcome_correct else 0.0,
            verification=verification,
            parsed=parsed,
        )
        return rv

    @staticmethod
    def _format_score(parsed: ParsedSolution) -> float:
        """Score structural compliance (think tags, steps, final result)."""
        score = 0.0
        if parsed.has_think_open:
            score += 0.25
        if parsed.has_think_close:
            score += 0.25
        if len(parsed.steps) > 0:
            score += 0.25
        if parsed.has_final_result_line:
            score += 0.25
        return score

    @staticmethod
    def _process_score(verification: SolutionVerification) -> float:
        """Fraction of steps with correct arithmetic (error-propagation-aware)."""
        return verification.computational_correctness

    @staticmethod
    def _consistency_score(parsed: ParsedSolution) -> float:
        """Fraction of steps with consistent claimed results vs expression updates."""
        if not parsed.steps:
            return 0.0

        checks = 0
        consistent = 0
        for step in parsed.steps:
            if step.expression_now is None:
                continue
            checks += 1
            # The claimed_result should appear somewhere in the expression_now
            # string.  We check for the string representation.
            if str(step.claimed_result) in step.expression_now:
                consistent += 1

        if checks == 0:
            # No "Expression now:" lines at all — mild penalty via 0.5
            return 0.5
        return consistent / checks
