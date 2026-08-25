# Error-Aware Process Supervision for Arithmetic GRPO

**STAT 359 Final Project — Northwestern University, March 2026**

Extends a decoder-only arithmetic LLM with *error-propagation-aware process supervision* and *adaptive reward scheduling* for GRPO. The core contribution is an `ExpressionStateTracker` that distinguishes independent arithmetic errors (wrong computation, 0 credit) from propagated errors (valid arithmetic on a wrong inherited value, 0.5 credit), replacing the standard binary outcome reward with a four-component signal.

Model output format:
```
<think>
Step 1: 10 - 3 = 7   |  Expression now: 5 + 7
Step 2: 5 + 7 = 12   |  Expression now: 12
</think>
Final Result: 12
```

---

## Setup

Requires Python 3.10+. Device auto-detected (CUDA → MPS → CPU).

```bash
git clone https://github.com/juanyeffrey/359_Final_Project_Testing.git
cd 359_Final_Project_Testing
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

All commands must be run from the repo root with the venv active.

---

## Training Pipeline

### Phase 1 — Foundational Pre-training

```bash
python -m arithmetic_llm.generate_foundational_plaintext \
    --num-samples 100000 --max-depth 4 --num-range 1 20 \
    --invalid-rate 0.05 --output-txt data/foundational_corpus.txt

python -m arithmetic_llm.train_tokenizer \
    --corpus-path data/foundational_corpus.txt \
    --output-dir data/tokenizer --vocab-size 1000

python -m arithmetic_llm.run_foundational_training \
    --corpus-path data/foundational_corpus.txt \
    --tokenizer-path data/tokenizer --output-dir models/ \
    --num-epochs 10 --batch-size 16
```

### Phase 2 — Instruction SFT

```bash
python -m arithmetic_llm.generate_instruction_corpus_mixed \
    --num-samples 20000 --max-depth 4 --num-range 1 20 \
    --invalid-rate 0 --output-mixed data/instruction_corpus.txt

python -m arithmetic_llm.run_instruction_training \
    --instruction-corpus-path data/instruction_corpus.txt \
    --tokenizer-path data/tokenizer \
    --foundational-checkpoint models/foundational_*/best_model.pt \
    --output-dir models/ --num-epochs 10
```

### Phase 3 — GRPO Experiment

Runs all four reward conditions from a shared SFT checkpoint, evaluates each, and saves consolidated results:

```bash
python -m arithmetic_llm.run_experiment \
    --tokenizer data/tokenizer \
    --sft-checkpoint models/instruction_*/best_model.pt \
    --output-dir experiments/run_01 \
    --num-samples 2000 --num-epochs 3 --batch-size 2 --eval-samples 500
```

To run a subset: `--conditions outcome_only error_aware`

Or run a single condition with custom reward settings:

```bash
python -m arithmetic_llm.run_grpo_training \
    --tokenizer data/tokenizer \
    --sft-checkpoint models/instruction_*/best_model.pt \
    --output-dir models/grpo_error_aware \
    --reward-mode error_aware --reward-weights 0.10 0.40 0.20 0.30
```

### Generate Figures

```bash
python -m arithmetic_llm.plot_results experiments/run_01
```

Or use the reproduction scripts in `scripts/`:
- `reproduce_full_pipeline.sh` — full Phase 1→2→3 from scratch
- `reproduce_experiment.sh <SFT_CKPT>` — Phase 3 only
- `regenerate_plots.sh` — figures only from saved results

---

## CLI Reference

All scripts support `--help`. Key flags:

| Script | Key flags |
|--------|-----------|
| `run_experiment.py` | `--tokenizer` `--sft-checkpoint` `--output-dir` `--num-samples` `--num-epochs` `--batch-size` `--eval-samples` `--conditions` `--seed` |
| `run_grpo_training.py` | same + `--reward-mode {outcome_only,naive_process,error_aware,scheduled}` `--reward-weights F F F F` `--schedule-strategy {linear,cosine,threshold,fixed}` |
| `run_evaluation.py` | `--model-path` `--tokenizer-path` `--num-samples` `--max-depth` `--num-range` |
| `plot_results.py` | `EXPERIMENT_DIR` (positional) |
| `run_interactive.py` | `--model-path` `--tokenizer-path` |

**Reward weights** are `[format, process, consistency, outcome]`. Default: `0.10 0.40 0.20 0.30`.

**Scheduler phases** (for `--reward-mode scheduled`): Phase 1 emphasizes format, Phase 2 process, Phase 3 outcome. Transitions controlled by `--schedule-phase1-frac` and `--schedule-phase2-frac`.

---

## Repository Structure

```
arithmetic_llm/
├── Core (course-provided)
│   ├── transformer_model.py        # 5.2M-param decoder-only transformer
│   ├── arithmetic_tokenizer.py     # BPE tokenizer, vocab=1000
│   ├── generator.py / evaluator.py # Expression generation and AST evaluation
│   ├── grpo_trainer.py             # GRPO training loop
│   └── train_*.py / run_*.py       # Training and CLI entry points
│
└── Novel contributions [NEW]
    ├── step_parser.py              # Model text → ParsedStep / ParsedSolution
    ├── expression_state_tracker.py # Dual-state error-propagation verifier
    ├── reward_decomposer.py        # Four-component RewardVector
    ├── reward_scheduler.py         # Adaptive weight scheduler (linear/cosine/threshold)
    ├── reward_functions.py         # Reward class hierarchy + build_reward_function()
    ├── extended_evaluator.py       # Reasoning taxonomy + per-depth breakdown
    ├── run_experiment.py           # Four-condition experiment orchestrator
    └── plot_results.py             # Publication figure generation

configs/                            # Annotated experiment config templates
scripts/                            # Shell scripts for one-command reproduction
data/tokenizer/                     # Pre-trained BPE tokenizer
experiments/run_01/                 # Results: JSONs, logs, plots (checkpoints excluded)
```

---

## Results

Trained 3,000 steps from the same SFT checkpoint on 2,000 generated problems (`max_depth=5`, `num_range=1–20`). Evaluated on 500 held-out samples (100 per depth).

### In-Distribution Accuracy

| Condition | Overall | Depth 4 | Depth 5 |
|-----------|---------|---------|---------|
| Outcome Only | 83.0% | 78% | 42% |
| Naïve Process | 83.2% | 78% | 43% |
| **Error Aware** | **83.6%** | **83%** | **45%** |
| Scheduled | **83.6%** | 82% | 44% |

### Propagated Error Reduction

| Condition | Propagated | Independent |
|-----------|------------|-------------|
| Outcome Only | 5 | 26 |
| Naïve Process | 4 | 23 |
| **Error Aware** | **1** | **20** |
| Scheduled | 2 | 19 |

**80% reduction in propagated errors** (5 → 1) confirms the partial-credit gradient signal works as intended. Overall accuracy differences are small due to a ceiling effect — the SFT checkpoint already achieves ~83% before GRPO begins, leaving little room for reward differentiation.

**OOD:** All conditions collapse to ~2% on numbers outside the training range. Root cause is the tokenizer — BPE trained only on 1–20 has no subword units for 3-digit numbers.

---

## Team Contributions

**Jeffrey Yuan** — Core process supervision system: `step_parser.py`, `expression_state_tracker.py`, `reward_decomposer.py`; integration of reward functions into `GRPOTrainer` and `run_grpo_training.py`; writeup and report.

**Yumeng Zhou** — Project proposal and design; adaptive curriculum system: `reward_scheduler.py`, `reward_functions.py`; evaluation and analysis pipeline: `extended_evaluator.py`, `run_experiment.py`, `plot_results.py`; training runs and qualitative error analysis.

Base codebase (`transformer_model.py`, tokenizer, generator, evaluator, Phase 1/2 training loops) provided by course instructors and used unmodified.

---

## License

MIT — see [LICENSE](LICENSE). Novel components are original work by Jeffrey Yuan and Yumeng Zhou. Base codebase provided by STAT 359 course instructors, Northwestern University.
