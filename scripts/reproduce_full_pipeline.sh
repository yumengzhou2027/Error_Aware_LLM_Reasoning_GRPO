#!/usr/bin/env bash
# reproduce_full_pipeline.sh
#
# Runs the complete training pipeline from scratch:
#   Phase 1: Generate data → train tokenizer → foundational pretraining
#   Phase 2: Generate instruction corpus → instruction SFT
#   Phase 3: 4-condition GRPO experiment → evaluation → plots
#
# Estimated runtime: ~6–8 hours on a GPU, ~30+ hours on CPU/MPS.
#
# Usage:
#   bash scripts/reproduce_full_pipeline.sh
#
# All output goes to data/, models/, and experiments/run_01/.

set -euo pipefail

echo "============================================"
echo " PHASE 1: Foundational Pre-training"
echo "============================================"

echo "--> Generating foundational corpus (100k samples)..."
python -m arithmetic_llm.generate_foundational_plaintext \
    --num-samples 100000 \
    --max-depth 4 \
    --num-range 1 20 \
    --invalid-rate 0.05 \
    --output-txt data/foundational_corpus.txt

echo "--> Training BPE tokenizer (vocab=1000)..."
python -m arithmetic_llm.train_tokenizer \
    --corpus-path data/foundational_corpus.txt \
    --output-dir data/tokenizer \
    --vocab-size 1000

echo "--> Foundational pretraining (10 epochs)..."
python -m arithmetic_llm.run_foundational_training \
    --corpus-path data/foundational_corpus.txt \
    --tokenizer-path data/tokenizer \
    --output-dir models/ \
    --num-epochs 10 \
    --batch-size 16

echo ""
echo "============================================"
echo " PHASE 2: Instruction SFT"
echo "============================================"

echo "--> Generating instruction corpus (20k samples)..."
python -m arithmetic_llm.generate_instruction_corpus_mixed \
    --num-samples 20000 \
    --max-depth 4 \
    --num-range 1 20 \
    --invalid-rate 0 \
    --output-mixed data/instruction_corpus.txt

echo "--> Instruction fine-tuning (10 epochs)..."
python -m arithmetic_llm.run_instruction_training \
    --instruction-corpus-path data/instruction_corpus.txt \
    --tokenizer-path data/tokenizer \
    --foundational-checkpoint models/foundational_*/best_model.pt \
    --output-dir models/ \
    --num-epochs 10

echo ""
echo "============================================"
echo " PHASE 3: GRPO Experiment (4 conditions)"
echo "============================================"

SFT_CKPT=$(ls -d models/instruction_*/best_model.pt | tail -1)
echo "--> Using SFT checkpoint: $SFT_CKPT"

python -m arithmetic_llm.run_experiment \
    --tokenizer data/tokenizer \
    --sft-checkpoint "$SFT_CKPT" \
    --output-dir experiments/run_01 \
    --num-samples 2000 \
    --num-epochs 3 \
    --batch-size 2 \
    --eval-samples 500 \
    --seed 42

echo ""
echo "--> Generating figures..."
python -m arithmetic_llm.plot_results experiments/run_01

echo ""
echo "============================================"
echo " Done."
echo " Results : experiments/run_01/consolidated_results.json"
echo " Figures : experiments/run_01/plot_*.png"
echo "============================================"
