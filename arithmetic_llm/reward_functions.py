"""Pluggable reward functions for GRPO training.

Each reward function implements a common interface::

    reward_fn(generated_text: str, ground_truth: int, prompt: str) -> float

This module provides four concrete reward functions matching the four
experimental conditions:

1. ``OutcomeOnlyReward``   — binary 0/1 (the existing baseline)
2. ``NaiveProcessReward``  — per-step correctness without error propagation
3. ``ErrorAwareReward``    — error-propagation-aware multi-component reward
4. ``ScheduledReward``     — error-aware + adaptive weight scheduling

All reward functions also expose ``last_reward_vector`` for logging.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from .arithmetic_verifier import ArithmeticVerifier
from .reward_decomposer import RewardDecomposer, RewardVector
from .reward_scheduler import RewardScheduler, SchedulerConfig
from .step_parser import ParsedSolution, extract_expression_from_prompt, parse_solution


class BaseRewardFunction(ABC):
    """Interface that all reward functions implement."""

    @abstractmethod
    def compute_reward(
        self,
        generated_text: str,
        ground_truth: int,
        prompt: str = "",
    ) -> float:
        """Return a scalar reward in [0, 1]."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Short identifier for logging / config files."""
        ...


class OutcomeOnlyReward(BaseRewardFunction):
    """Binary 0/1 reward baseline using ArithmeticVerifier."""

    def __init__(self) -> None:
        self._verifier = ArithmeticVerifier()
        self.last_reward_vector: Optional[RewardVector] = None

    def compute_reward(
        self,
        generated_text: str,
        ground_truth: int,
        prompt: str = "",
    ) -> float:
        reward = self._verifier.compute_reward(generated_text, ground_truth)
        # Build a minimal RewardVector for logging uniformity
        self.last_reward_vector = RewardVector(
            format_score=0.0,
            process_score=0.0,
            consistency_score=0.0,
            outcome_score=reward,
        )
        return reward

    def name(self) -> str:
        return "outcome_only"


class NaiveProcessReward(BaseRewardFunction):
    """Per-step correctness reward without error-propagation awareness."""

    def __init__(
        self,
        weights: Tuple[float, float, float, float] = (0.10, 0.40, 0.20, 0.30),
    ) -> None:
        self._weights = weights
        self.last_reward_vector: Optional[RewardVector] = None

    def compute_reward(
        self,
        generated_text: str,
        ground_truth: int,
        prompt: str = "",
    ) -> float:
        parsed = parse_solution(generated_text)

        # Naive step accuracy: check each A op B = C in isolation
        if parsed.steps:
            correct = 0
            for step in parsed.steps:
                a, op, b, c = (
                    step.left_operand,
                    step.operator,
                    step.right_operand,
                    step.claimed_result,
                )
                expected = a + b if op == "+" else a - b
                if expected == c:
                    correct += 1
            naive_process = correct / len(parsed.steps)
        else:
            naive_process = 0.0

        outcome = 1.0 if parsed.final_result == ground_truth else 0.0

        # Format score (reuse the same logic)
        fmt = 0.0
        if parsed.has_think_open:
            fmt += 0.25
        if parsed.has_think_close:
            fmt += 0.25
        if parsed.steps:
            fmt += 0.25
        if parsed.has_final_result_line:
            fmt += 0.25

        # Consistency (same as RewardDecomposer)
        if parsed.steps:
            checks = 0
            consistent = 0
            for step in parsed.steps:
                if step.expression_now is None:
                    continue
                checks += 1
                if str(step.claimed_result) in step.expression_now:
                    consistent += 1
            consistency = (consistent / checks) if checks > 0 else 0.5
        else:
            consistency = 0.0

        rv = RewardVector(
            format_score=fmt,
            process_score=naive_process,
            consistency_score=consistency,
            outcome_score=outcome,
            parsed=parsed,
        )
        self.last_reward_vector = rv
        return rv.to_scalar(self._weights)

    def name(self) -> str:
        return "naive_process"


class ErrorAwareReward(BaseRewardFunction):
    """Error-propagation-aware multi-component reward with fixed weights."""

    def __init__(
        self,
        weights: Tuple[float, float, float, float] = (0.10, 0.40, 0.20, 0.30),
    ) -> None:
        self._weights = weights
        self._decomposer = RewardDecomposer()
        self.last_reward_vector: Optional[RewardVector] = None

    def compute_reward(
        self,
        generated_text: str,
        ground_truth: int,
        prompt: str = "",
    ) -> float:
        expression = extract_expression_from_prompt(prompt) if prompt else ""
        rv = self._decomposer.compute(generated_text, expression, ground_truth)
        self.last_reward_vector = rv
        return rv.to_scalar(self._weights)

    def name(self) -> str:
        return "error_aware"


class ScheduledReward(BaseRewardFunction):
    """Error-aware reward with adaptive weight scheduling."""

    def __init__(self, scheduler_config: SchedulerConfig) -> None:
        self._decomposer = RewardDecomposer()
        self._scheduler = RewardScheduler(scheduler_config)
        self._current_step: int = 0
        self.last_reward_vector: Optional[RewardVector] = None

    @property
    def scheduler(self) -> RewardScheduler:
        return self._scheduler

    def compute_reward(
        self,
        generated_text: str,
        ground_truth: int,
        prompt: str = "",
    ) -> float:
        expression = extract_expression_from_prompt(prompt) if prompt else ""
        rv = self._decomposer.compute(generated_text, expression, ground_truth)
        self.last_reward_vector = rv

        weights = self._scheduler.get_weights(self._current_step)
        return rv.to_scalar(weights)

    def step(self, format_score: float = 0.0, process_score: float = 0.0) -> None:
        """Advance the scheduler by one training step.

        Parameters
        ----------
        format_score, process_score : float
            Batch-average scores for the threshold strategy's EMA tracker.
        """
        self._scheduler.update_metrics(
            format_score=format_score,
            process_score=process_score,
        )
        self._current_step += 1

    def current_phase(self) -> int:
        return self._scheduler.current_phase(self._current_step)

    def current_weights(self) -> Tuple[float, float, float, float]:
        return self._scheduler.get_weights(self._current_step)

    def name(self) -> str:
        return f"scheduled_{self._scheduler.config.strategy}"


def build_reward_function(
    mode: str,
    weights: Tuple[float, float, float, float] = (0.10, 0.40, 0.20, 0.30),
    scheduler_config: Optional[SchedulerConfig] = None,
) -> BaseRewardFunction:
    """Instantiate a reward function by name."""
    if mode == "outcome_only":
        return OutcomeOnlyReward()
    elif mode == "naive_process":
        return NaiveProcessReward(weights=weights)
    elif mode == "error_aware":
        return ErrorAwareReward(weights=weights)
    elif mode == "scheduled":
        if scheduler_config is None:
            raise ValueError("scheduler_config is required for 'scheduled' mode")
        return ScheduledReward(scheduler_config=scheduler_config)
    else:
        raise ValueError(
            f"Unknown reward mode: {mode!r}.  "
            f"Choose from: outcome_only, naive_process, error_aware, scheduled."
        )
