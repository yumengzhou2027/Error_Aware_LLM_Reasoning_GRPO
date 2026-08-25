"""Adaptive reward weight scheduling for multi-component GRPO rewards.

Shifts emphasis across training phases:
  - **Phase 1 (Format)**: High weight on format compliance to establish structure
  - **Phase 2 (Process)**: Shift toward process correctness once format is stable
  - **Phase 3 (Outcome)**: Final emphasis on correct answers

Three scheduling strategies are provided:
  - ``linear``    : smooth linear interpolation between phase boundaries
  - ``cosine``    : cosine-annealed transitions (smoother at boundaries)
  - ``threshold`` : hard phase transitions triggered by running-average metrics
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class SchedulerConfig:
    """Configuration for the reward weight scheduler."""

    strategy: str = "linear"
    total_steps: int = 1000

    phase1_frac: float = 0.25
    phase2_frac: float = 0.50

    # (format, process, consistency, outcome)
    phase1_weights: Tuple[float, float, float, float] = (0.50, 0.20, 0.20, 0.10)
    phase2_weights: Tuple[float, float, float, float] = (0.10, 0.40, 0.20, 0.30)
    phase3_weights: Tuple[float, float, float, float] = (0.05, 0.15, 0.10, 0.70)

    # Threshold-strategy knobs
    threshold_format: float = 0.85
    threshold_process: float = 0.60
    ema_alpha: float = 0.05

    def validate(self) -> None:
        if self.strategy not in ("linear", "cosine", "threshold", "fixed"):
            raise ValueError(f"Unknown strategy: {self.strategy!r}")
        if self.total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {self.total_steps}")
        if not (0 <= self.phase1_frac <= 1):
            raise ValueError(f"phase1_frac must be in [0, 1], got {self.phase1_frac}")
        if not (0 <= self.phase2_frac <= 1):
            raise ValueError(f"phase2_frac must be in [0, 1], got {self.phase2_frac}")
        if self.phase1_frac + self.phase2_frac > 1.0:
            raise ValueError(
                f"phase1_frac + phase2_frac must be <= 1, "
                f"got {self.phase1_frac} + {self.phase2_frac}"
            )

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "total_steps": self.total_steps,
            "phase1_frac": self.phase1_frac,
            "phase2_frac": self.phase2_frac,
            "phase1_weights": list(self.phase1_weights),
            "phase2_weights": list(self.phase2_weights),
            "phase3_weights": list(self.phase3_weights),
            "threshold_format": self.threshold_format,
            "threshold_process": self.threshold_process,
            "ema_alpha": self.ema_alpha,
        }


class RewardScheduler:
    """Return time-varying reward weights based on training progress."""

    def __init__(self, config: SchedulerConfig):
        config.validate()
        self.config = config
        self._phase1_end = int(config.total_steps * config.phase1_frac)
        self._phase2_end = int(config.total_steps * (config.phase1_frac + config.phase2_frac))

        # EMA trackers (for threshold strategy)
        self._ema_format: float = 0.0
        self._ema_process: float = 0.0
        self._threshold_phase: int = 1  # current phase for threshold strategy

    def get_weights(self, step: int) -> Tuple[float, float, float, float]:
        """Return (w_format, w_process, w_consistency, w_outcome) for the given step."""
        strategy = self.config.strategy
        if strategy == "fixed":
            return self.config.phase2_weights  # sensible default for fixed
        elif strategy == "linear":
            return self._linear(step)
        elif strategy == "cosine":
            return self._cosine(step)
        elif strategy == "threshold":
            return self._threshold()
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

    def update_metrics(
        self,
        format_score: float = 0.0,
        process_score: float = 0.0,
    ) -> None:
        """Feed per-step metrics for the threshold strategy."""
        alpha = self.config.ema_alpha
        self._ema_format = alpha * format_score + (1 - alpha) * self._ema_format
        self._ema_process = alpha * process_score + (1 - alpha) * self._ema_process

        # Phase transitions are one-directional (never go backward)
        if self._threshold_phase == 1 and self._ema_format >= self.config.threshold_format:
            self._threshold_phase = 2
        if self._threshold_phase == 2 and self._ema_process >= self.config.threshold_process:
            self._threshold_phase = 3

    def current_phase(self, step: int) -> int:
        """Return the current phase number (1, 2, or 3)."""
        if self.config.strategy == "threshold":
            return self._threshold_phase
        if step < self._phase1_end:
            return 1
        if step < self._phase2_end:
            return 2
        return 3

    def state_dict(self) -> dict:
        """Serialize scheduler state for checkpointing."""
        return {
            "ema_format": self._ema_format,
            "ema_process": self._ema_process,
            "threshold_phase": self._threshold_phase,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore scheduler state from checkpoint."""
        self._ema_format = state.get("ema_format", 0.0)
        self._ema_process = state.get("ema_process", 0.0)
        self._threshold_phase = state.get("threshold_phase", 1)

    def _lerp(
        self,
        w_a: Tuple[float, float, float, float],
        w_b: Tuple[float, float, float, float],
        t: float,
    ) -> Tuple[float, float, float, float]:
        """Linearly interpolate between two weight tuples."""
        t = max(0.0, min(1.0, t))
        return tuple(a + (b - a) * t for a, b in zip(w_a, w_b))  # type: ignore[return-value]

    def _linear(self, step: int) -> Tuple[float, float, float, float]:
        cfg = self.config
        if step < self._phase1_end:
            # Phase 1: interpolate from phase1 -> phase2 start
            t = step / max(self._phase1_end, 1)
            return self._lerp(cfg.phase1_weights, cfg.phase2_weights, t)
        elif step < self._phase2_end:
            # Phase 2: interpolate from phase2 -> phase3 start
            t = (step - self._phase1_end) / max(self._phase2_end - self._phase1_end, 1)
            return self._lerp(cfg.phase2_weights, cfg.phase3_weights, t)
        else:
            return cfg.phase3_weights

    def _cosine(self, step: int) -> Tuple[float, float, float, float]:
        """Cosine-annealed interpolation."""
        cfg = self.config
        if step < self._phase1_end:
            t = step / max(self._phase1_end, 1)
            t = 0.5 * (1 - math.cos(math.pi * t))  # cosine ease-in-out
            return self._lerp(cfg.phase1_weights, cfg.phase2_weights, t)
        elif step < self._phase2_end:
            t = (step - self._phase1_end) / max(self._phase2_end - self._phase1_end, 1)
            t = 0.5 * (1 - math.cos(math.pi * t))
            return self._lerp(cfg.phase2_weights, cfg.phase3_weights, t)
        else:
            return cfg.phase3_weights

    def _threshold(self) -> Tuple[float, float, float, float]:
        """Return weights for the current threshold phase."""
        cfg = self.config
        if self._threshold_phase == 1:
            return cfg.phase1_weights
        elif self._threshold_phase == 2:
            return cfg.phase2_weights
        else:
            return cfg.phase3_weights
