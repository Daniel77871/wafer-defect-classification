#!/usr/bin/env bash
set -euo pipefail

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON:-.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON:-python}"
fi

export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

run_python() {
  local script="$1"
  shift
  echo "Running ${script}"
  "${PYTHON_BIN}" "${script}" "$@"
}

run_experiment() {
  local config="$1"
  shift
  echo "Running ${config} with WAFER_DEVICE=${WAFER_DEVICE:-auto}"
  "${PYTHON_BIN}" scripts/run_experiment.py --config "${config}" "$@"
}

run_python scripts/run_eda.py
run_python scripts/run_preprocessing.py

run_experiment configs/exp_A_cnn_scratch.yaml "$@"
run_experiment configs/exp_B_transfer.yaml "$@"
run_experiment configs/exp_C_class_weight.yaml "$@"
run_experiment configs/exp_D_focal_loss.yaml "$@"
run_experiment configs/exp_E_augmentation.yaml "$@"
run_experiment configs/exp_F_svm.yaml "$@"

run_python scripts/run_gradcam_analysis.py
run_python scripts/generate_report_materials.py
