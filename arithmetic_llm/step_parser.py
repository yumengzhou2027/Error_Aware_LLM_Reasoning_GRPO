"""Parse model-generated arithmetic solutions into structured step data.

This module extracts reasoning steps, expression-now lines, and final results
from the model's free-form text output, producing dataclasses that downstream
verification and reward modules consume.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ParsedStep:
    """A single reasoning step extracted from generated text."""
    step_number: int
    left_operand: int
    operator: str
    right_operand: int
    claimed_result: int
    expression_now: Optional[str] = None
    char_span: Tuple[int, int] = (0, 0)


@dataclass
class ParsedSolution:
    """Full parsed representation of a model-generated solution."""
    steps: List[ParsedStep] = field(default_factory=list)
    final_result: Optional[int] = None
    has_think_open: bool = False
    has_think_close: bool = False
    has_final_result_line: bool = False
    raw_text: str = ""


_STEP_PATTERN = re.compile(
    r"Step\s+(\d+)\s*:\s*(-?\d+)\s*([+\-])\s*(-?\d+)\s*=\s*(-?\d+)"
)

_EXPR_NOW_PATTERN = re.compile(
    r"Expression\s+now\s*:\s*(.+)"
)

_FINAL_RESULT_PATTERN = re.compile(
    r"Final\s+Result\s*:\s*([+-]?\s*\d+)",
    re.IGNORECASE,
)

_FINAL_RESULT_ERROR_PATTERN = re.compile(
    r"Final\s+Result\s*:\s*ERROR",
    re.IGNORECASE,
)

_THINK_OPEN = re.compile(r"<think>")
_THINK_CLOSE = re.compile(r"</think>")


def parse_solution(generated_text: str) -> ParsedSolution:
    """Parse a model-generated solution into structured data."""
    result = ParsedSolution(raw_text=generated_text)

    result.has_think_open = bool(_THINK_OPEN.search(generated_text))
    result.has_think_close = bool(_THINK_CLOSE.search(generated_text))

    step_matches: List[Tuple[re.Match, ParsedStep]] = []
    for m in _STEP_PATTERN.finditer(generated_text):
        ps = ParsedStep(
            step_number=int(m.group(1)),
            left_operand=int(m.group(2)),
            operator=m.group(3),
            right_operand=int(m.group(4)),
            claimed_result=int(m.group(5)),
            char_span=(m.start(), m.end()),
        )
        step_matches.append((m, ps))

    expr_now_matches = list(_EXPR_NOW_PATTERN.finditer(generated_text))
    expr_idx = 0
    for i, (m, ps) in enumerate(step_matches):
        while expr_idx < len(expr_now_matches):
            en = expr_now_matches[expr_idx]
            if en.start() > m.end():
                next_step_start = (
                    step_matches[i + 1][0].start()
                    if i + 1 < len(step_matches)
                    else len(generated_text)
                )
                if en.start() < next_step_start:
                    ps.expression_now = en.group(1).strip()
                    expr_idx += 1
                break
            expr_idx += 1

    result.steps = [ps for _, ps in step_matches]

    if _FINAL_RESULT_ERROR_PATTERN.search(generated_text):
        result.has_final_result_line = True
        result.final_result = None
    else:
        fm = _FINAL_RESULT_PATTERN.search(generated_text)
        if fm:
            result.has_final_result_line = True
            try:
                result.final_result = int(fm.group(1).replace(" ", ""))
            except ValueError:
                result.final_result = None

    return result


def extract_expression_from_prompt(prompt: str) -> str:
    """Extract the raw arithmetic expression from an instruction prompt."""
    text = prompt
    if text.startswith("Evaluate:"):
        text = text[len("Evaluate:"):].strip()
    think_idx = text.rfind("<think>")
    if think_idx != -1:
        text = text[:think_idx].strip()
    return text
