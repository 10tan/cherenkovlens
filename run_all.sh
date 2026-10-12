#!/usr/bin/env bash
# ==============================================================================
# CherenkovLens Complete Pipeline Runner
# Executes:
# 1. Dataset generation
# 2. Cleaning comparison study
# 3. Hillas ground-truth & ctapipe validation
# 4. Classifier training & Q-factor optimization
# 5. Night Sky Background (NSB) noise robustness study
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo " Starting CherenkovLens End-to-End Pipeline"
echo "======================================================================"

echo ""
echo "[Step 1/5] Generating Primary Dataset (40,000 events)..."
python3 scripts/generate_dataset.py --n-events 40000 --gamma-fraction 0.5 --seed 42

echo ""
echo "[Step 2/5] Running Image Cleaning Study (Tail-Cuts vs Naive)..."
python3 scripts/run_cleaning_study.py --n-events 2000 --seed 42

echo ""
echo "[Step 3/5] Validating Hillas Extraction against Simulation Ground-Truth..."
python3 scripts/validate_hillas.py --n-events 1000 --seed 42

echo ""
echo "[Step 4/5] Training Gamma/Hadron Classifiers and Computing Q-Factor..."
python3 scripts/train_classifier.py --seed 42

echo ""
echo "[Step 5/5] Running NSB Noise Robustness Benchmark (1x, 2x, 4x, 8x NSB)..."
python3 scripts/run_noise_study.py --events-per-level 3000 --seed 42

echo ""
echo "======================================================================"
echo " CherenkovLens Pipeline Run Completed Successfully!"
echo " All figures saved to: figures/"
echo " All metrics saved to: results/"
echo " Model saved to:       models/classifier_rf.joblib"
echo "======================================================================"
