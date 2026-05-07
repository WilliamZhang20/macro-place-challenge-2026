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
#   TILUS_CONDA_ENV (default: myenv), TILUS_CONDA_BASE — same as hopper_matmul_benchmark.sh
#   CONDA_ENV — legacy alias for TILUS_CONDA_ENV if TILUS_CONDA_ENV is unset
#   TILUS_CUDA_MODULE — optional `module load` argument after sourcing modules.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"

mkdir -p tuning_logs

echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-}  Start: $(date)"
echo "Repo: ${REPO_ROOT}"

# ── Environment modules ────────────────────────────────────────────────────────
if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
fi
if [[ -n "${TILUS_CUDA_MODULE:-}" ]]; then
  module load "${TILUS_CUDA_MODULE}"
fi

# ── Conda (aligned with hopper_matmul_benchmark.sh) ───────────────────────────
: "${TILUS_CONDA_ENV:=${CONDA_ENV:-myenv}}"

if [[ -n "${TILUS_CONDA_ENV:-}" ]]; then
  _conda_sh=""
  if [[ -n "${TILUS_CONDA_BASE:-}" && -f "${TILUS_CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    _conda_sh="${TILUS_CONDA_BASE}/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    _conda_sh="${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    _conda_sh="${HOME}/anaconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/mambaforge/etc/profile.d/conda.sh" ]]; then
    _conda_sh="${HOME}/mambaforge/etc/profile.d/conda.sh"
  elif [[ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]]; then
    _conda_sh="/opt/anaconda3/etc/profile.d/conda.sh"
  elif command -v conda >/dev/null 2>&1; then
    _b="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${_b}" && -f "${_b}/etc/profile.d/conda.sh" ]]; then
      _conda_sh="${_b}/etc/profile.d/conda.sh"
    fi
  fi
  if [[ -z "${_conda_sh}" ]]; then
    echo "ERROR: could not find conda.sh. Set TILUS_CONDA_BASE to your conda install root." >&2
    exit 4
  fi
  # shellcheck source=/dev/null
  set +u
  source "${_conda_sh}"
  conda activate "${TILUS_CONDA_ENV}"
  set -u
fi

# ── CUDA / nvcc ────────────────────────────────────────────────────────────────
for p in \
    "${CONDA_PREFIX:-}/bin" \
    "${CUDA_HOME:-}/bin" \
    "${CUDA_PATH:-}/bin" \
    "/usr/local/cuda/bin"; do
  [[ -x "${p}/nvcc" ]] && export PATH="${p}:${PATH}"
done

if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found. GPU DREAMPlace needs the CUDA toolkit on PATH." >&2
  exit 3
fi
echo "nvcc: $(command -v nvcc)  ($(nvcc --version 2>/dev/null | grep release | head -1 || true))"

# ── Python path ────────────────────────────────────────────────────────────────
export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:$PYTHONPATH}"

# Prefer interpreter from the activated env when possible.
PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" && -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON="${CONDA_PREFIX}/bin/python"
fi
if [[ -z "${PYTHON}" ]]; then
  for _cand in "${HOME}/.conda/envs/${TILUS_CONDA_ENV}/bin/python" \
               "${HOME}/miniconda3/envs/${TILUS_CONDA_ENV}/bin/python" \
               "${HOME}/anaconda3/envs/${TILUS_CONDA_ENV}/bin/python"; do
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
