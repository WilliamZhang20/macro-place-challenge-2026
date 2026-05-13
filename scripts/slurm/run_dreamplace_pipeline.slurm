#!/bin/bash
# Run the DREAMPlace -> RePlAce refinement pipeline evaluator.
#
# Submit from repo root:
#   sbatch scripts/slurm/run_dreamplace_pipeline.slurm
#
# Optional:
#   sbatch --export=ALL,EVAL_ARGS="-b ibm01" scripts/slurm/run_dreamplace_pipeline.slurm
#   sbatch --export=ALL,MACRO_PLACE_DP_CONFIG=tuning_logs/dreamplace_optuna_best.json scripts/slurm/run_dreamplace_pipeline.slurm
#   sbatch --export=ALL,TILUS_CONDA_ENV=myenv,TILUS_CONDA_BASE=$HOME/miniconda3 scripts/slurm/run_dreamplace_pipeline.slurm
#
#SBATCH --job-name=dreamplace_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=sweep_logs/slurm_%j_%x.out
#SBATCH --error=sweep_logs/slurm_%j_%x.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"

mkdir -p sweep_logs

echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-}  Start: $(date)"
echo "Repo: ${REPO_ROOT}"

# -- Environment modules -------------------------------------------------------
if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
fi
if [[ -n "${TILUS_CUDA_MODULE:-}" ]]; then
  module load "${TILUS_CUDA_MODULE}"
fi

# -- Conda --------------------------------------------------------------------
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

# -- CUDA / nvcc ---------------------------------------------------------------
for p in \
    "${CONDA_PREFIX:-}/bin" \
    "${CUDA_HOME:-}/bin" \
    "${CUDA_PATH:-}/bin" \
    "/usr/local/cuda/bin"; do
  [[ -x "${p}/nvcc" ]] && export PATH="${p}:${PATH}"
done

if command -v nvcc >/dev/null 2>&1; then
  echo "nvcc: $(command -v nvcc)  ($(nvcc --version 2>/dev/null | grep release | head -1 || true))"
else
  echo "WARNING: nvcc not found on PATH; DREAMPlace will only work if this install can run without it." >&2
fi

# -- Python path ---------------------------------------------------------------
export PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}/submissions${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MACRO_PLACE_DP_GPU="${MACRO_PLACE_DP_GPU:-1}"
export MACRO_PLACE_TUNER_DEBUG="${MACRO_PLACE_TUNER_DEBUG:-1}"

PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" && -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON="${CONDA_PREFIX}/bin/python"
fi
PYTHON="${PYTHON:-python}"

PLACER="${PLACER:-submissions/dreamplace_pipeline_placer.py}"
EVAL_ARGS="${EVAL_ARGS:---all}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-sweep_logs/dreamplace_pipeline_${SLURM_JOB_ID:-local}_${STAMP}.log}"

echo "python: $(${PYTHON} -c 'import sys; print(sys.executable)')"
"${PYTHON}" - <<'PY'
from submissions._dreamplace_cpu_smoke import default_dreamplace_install, dreamplace_install_ok
p = default_dreamplace_install()
ok, msg = dreamplace_install_ok(p)
print(f"DREAMPlace install: {p}  ok={ok}  msg={msg}")
if not ok:
    raise SystemExit(5)
PY
echo "MACRO_PLACE_DP_CONFIG=${MACRO_PLACE_DP_CONFIG:-}"
echo "MACRO_PLACE_DP_GPU=${MACRO_PLACE_DP_GPU}"
echo "Running: ${PYTHON} -m macro_place.evaluate ${PLACER} ${EVAL_ARGS}"
echo "Log: ${LOG_FILE}"

RUNNER=()
if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
  RUNNER=(srun)
fi

"${RUNNER[@]}" "${PYTHON}" -m macro_place.evaluate "${PLACER}" ${EVAL_ARGS} 2>&1 | tee "${LOG_FILE}"

echo "Done: $(date)"
