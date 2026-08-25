#!/usr/bin/env bash
# reproduce_experiment.sh
#
# Reproduces the full 4-condition GRPO experiment from scratch.
# Assumes you have already completed Phases 1 and 2 (foundational pretraining
# and instruction SFT). If not, run reproduce_full_pipeline.sh instead.
#
# Usage:
#   bash scripts/reproduce_experiment.sh <SFT_CHECKPOINT> [OUTPUT_DIR]
#
# Example:
#   bash scripts/reproduce_experiment.sh \
#       models/instruction_20260220_131538_745585/best_model.pt \
#       experiments/run_01
#
# Output:
#   <OUTPUT_DIR>/consolidated_results.json  — all metrics
#   <OUTPUT_DIR>/plot_*.png                 — publication figures

set -euo pipefail

SFT_CHECKPOINT="${1:?Usage: $0 <SFT_CHECKPOINT> [OUTPUT_DIR]}"
OUTPUT_DIR="${2:-experiments/run_01}"

echo "==> Running 4-condition GRPO experiment"
echo "    SFT checkpoint : $SFT_CHECKPOINT"
echo "    Output dir     : $OUTPUT_DIR"
echo ""

python -m arithmetic_llm.run_experiment \
    --tokenizer data/tokenizer \
    --sft-checkpoint "$SFT_CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --num-samples 2000 \
    --num-epochs 3 \
    --batch-size 2 \
    --eval-samples 500 \
    --seed 42

echo ""
echo "==> Generating plots"
python -m arithmetic_llm.plot_results "$OUTPUT_DIR"

echo ""
echo "==> Done. Results in $OUTPUT_DIR/"
echo "    Figures: $OUTPUT_DIR/plot_*.png"
echo "    Metrics: $OUTPUT_DIR/consolidated_results.json"
