# arithmetic_llm -- Module Architecture

For usage instructions, see the [top-level README](../README.md).

## Module Map

### Core Pipeline

| Module | Role |
|---|---|
| `transformer_model.py` | `ArithmeticTransformer` -- decoder-only, ~5.2M params (d_model=256, 6 layers, 8 heads) |
| `arithmetic_tokenizer.py` | BPE tokenizer (vocab=1000). Special tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>`, `<think>`, `</think>` |
| `generator.py` | `ExpressionGenerator` -- random expression trees (+ and - with parens) |
| `evaluator.py` | `eval_expression()` -- AST-based evaluation producing step-by-step solutions |
| `data_loader.py` | `ArithmeticDataset` -- loads JSONL corpora for foundational/instruction training |
| `arithmetic_verifier.py` | `ArithmeticVerifier` -- binary reward (final answer correct?) |

### Training Phases

| Module | Phase |
|---|---|
| `train_foundational.py` | Phase 1: next-token prediction on full expression+solution text |
| `train_instruction.py` | Phase 2: instruction SFT with prompt-masked loss |
| `train_instruction_lora.py` | Phase 2 (LoRA variant): low-rank adapters on attention projections |
| `grpo_trainer.py` | Phase 3: GRPO training loop with pluggable reward functions |
| `grpo_config.py` | `GRPOConfig` dataclass with all training + reward hyperparameters |
| `train_grpo.py` | Phase 3 entry point, builds reward function from config |

### Process Supervision (Final Project Extension)

| Module | Role |
|---|---|
| `step_parser.py` | Regex-based parser: generated text -> `ParsedStep` / `ParsedSolution` |
| `expression_state_tracker.py` | Error-propagation-aware verifier with tainted value tracking |
| `reward_decomposer.py` | `RewardVector` (format/process/consistency/outcome) + `RewardDecomposer` |
| `reward_scheduler.py` | `RewardScheduler` with linear/cosine/threshold/fixed weight strategies |
| `reward_functions.py` | Four reward classes + `build_reward_function()` factory |
| `extended_evaluator.py` | Reasoning taxonomy (correct/specious/unlucky/failed) + per-depth breakdown |
| `run_experiment.py` | Orchestrates all 4 experimental conditions |
| `plot_results.py` | 7 publication-quality matplotlib figures |

## Data Flow

```
ExpressionGenerator.generate()
    -> eval_expression() -> step-by-step solution with <think> tags
    -> JSONL corpus
    -> ArithmeticDataset -> training
```

## Model Output Format

```
<think>
Step 1: 10 - 3 = 7
Expression now: 5 + 7
Step 2: 5 + 7 = 12
Expression now: 12
</think>
Final Result: 12
```
