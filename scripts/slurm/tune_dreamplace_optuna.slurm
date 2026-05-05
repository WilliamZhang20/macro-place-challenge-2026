#!/bin/bash
#SBATCH --job-name=dp_optuna
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=tuning_logs/slurm_%j_%x.out
#SBATCH --error=tuning_logs/slurm_%j_%x.err

# DREAMPlace + Optuna hyperparameter search (one GPU, any CUDA device).
# Submit from the repository root (or set REPO_ROOT).
#
#   sbatch scripts/slurm/tune_dreamplace_optuna.slurm
#   sbatch --export=ALL,N_TRIALS=300,SEED=1 scripts/slurm/tune_dreamplace_optuna.slurm
#
# Optional env:
#   N_TRIALS, SEED, BENCHMARKS (comma list; empty = script default calibration set)
#   CONDA_ENV (default: myenv) — `conda activate` is attempted when conda is available;
#     otherwise falls back to `~/.conda/envs/$CONDA_ENV/bin/python` if present.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"

mkdir -p tuning_logs

CONDA_ENV="${CONDA_ENV:-myenv}"
if [[ -f "${HOME}/.conda/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/.conda/etc/profile.d/conda.sh"
elif [[ -f "/u5/w223zhan/.conda/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "/u5/w223zhan/.conda/etc/profile.d/conda.sh"
elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "/opt/conda/etc/profile.d/conda.sh"
fi

if command -v conda >/dev/null 2>&1; then
  conda activate "${CONDA_ENV}"
fi

# Prefer interpreter from the activated env when possible.
PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" && -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON="${CONDA_PREFIX}/bin/python"
fi
if [[ -z "${PYTHON}" ]]; then
  for _cand in "${HOME}/.conda/envs/${CONDA_ENV}/bin/python" \
               "/u5/w223zhan/.conda/envs/${CONDA_ENV}/bin/python"; do
    if [[ -x "${_cand}" ]]; then
      PYTHON="${_cand}"
      break
    fi
  done
fi
PYTHON="${PYTHON:-python}"
N_TRIALS="${N_TRIALS:-200}"
SEED="${SEED:-0}"

export MACRO_PLACE_DP_GPU="${MACRO_PLACE_DP_GPU:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

CMD=( "${PYTHON}" scripts/tune_dreamplace_optuna.py --n-trials "${N_TRIALS}" --seed "${SEED}" --multivariate-tpe )
if [[ -n "${BENCHMARKS:-}" ]]; then
  CMD+=( --benchmarks "${BENCHMARKS}" )
fi

echo "REPO_ROOT=${REPO_ROOT}"
echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
