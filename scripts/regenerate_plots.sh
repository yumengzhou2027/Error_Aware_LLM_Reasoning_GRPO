#!/usr/bin/env bash
# regenerate_plots.sh
#
# Regenerates all publication figures from existing experiment results.
# Does not require re-running training — reads from consolidated_results.json.
#
# Usage:
#   bash scripts/regenerate_plots.sh [EXPERIMENT_DIR]
#
# Example:
#   bash scripts/regenerate_plots.sh experiments/run_01

set -euo pipefail

EXPERIMENT_DIR="${1:-experiments/run_01}"

if [ ! -f "$EXPERIMENT_DIR/consolidated_results.json" ]; then
    echo "ERROR: $EXPERIMENT_DIR/consolidated_results.json not found."
    echo "Run the experiment first: bash scripts/reproduce_experiment.sh <SFT_CHECKPOINT>"
    exit 1
fi

echo "==> Generating plots from $EXPERIMENT_DIR"
python -m arithmetic_llm.plot_results "$EXPERIMENT_DIR"

echo ""
echo "==> Figures saved to $EXPERIMENT_DIR/:"
ls "$EXPERIMENT_DIR"/plot_*.png 2>/dev/null || echo "  (none found)"
