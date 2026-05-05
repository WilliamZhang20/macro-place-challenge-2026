#!/usr/bin/env bash
# Build and install DREAMPlace under external/DREAMPlace/install so submissions can run:
#   cd external/DREAMPlace/install && python dreamplace/Placer.py <config.json>
#
# Prereqs (typical Ubuntu/WSL):
#   - cmake, make, flex, bison, libboost-dev, zlib, OpenMP (build-essential)
#   - PyTorch in the *same* Python you pass (default: current `python3` on PATH)
#
# Optional:
#   PYTHON=/path/to/venv/bin/python   # use the env that has torch (e.g. ~/myenv)
#   CMAKE_BUILD_PARALLEL_LEVEL=2     # WSL: avoid -j$(nproc) OOM; default 2
#   CMAKE_CXX_ABI=0|1                 # must match PyTorch; default: auto from torch
#   SKIP_PIP=1                        # skip requirements.txt install
#   FORCE_CLEAN=1                     # delete build/ first (full reconfigure)
#   FORCE_RECONFIGURE=1               # re-run cmake even if CMakeCache.txt exists
#   CUDA: if nvcc is on PATH, re-run cmake and it will pick up GPU; you can set
#         CMAKE_CUDA_FLAGS=-gencode=arch=compute_80,code=sm_80 for your GPU.
#   At run time, submissions honor MACRO_PLACE_DP_GPU=0|1|auto (see
#   submissions/_dreamplace_cpu_smoke.resolve_dreamplace_gpu).
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DP="${ROOT_DIR}/external/DREAMPlace"
BUILD="${DP}/build"
INSTALL="${DP}/install"
JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"

if [[ ! -d "${DP}/dreamplace" ]]; then
  echo "ERROR: ${DP} missing or incomplete. Run: git submodule update --init external/DREAMPlace" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

PYTHON_ABS="$(command -v "${PYTHON_BIN}")"
echo "Using Python: ${PYTHON_ABS}"
"${PYTHON_ABS}" -c "import torch" 2>/dev/null || {
  echo "ERROR: PyTorch not importable with this interpreter. Activate your venv or set PYTHON=" >&2
  exit 1
}

echo "Initializing DREAMPlace third-party submodules..."
git -C "${DP}" submodule update --init --recursive

if [[ "${SKIP_PIP:-0}" != "1" ]]; then
  echo "Installing Python deps from DREAMPlace/requirements.txt ..."
  "${PYTHON_ABS}" -m pip install -q -r "${DP}/requirements.txt"
fi

if [[ -z "${CMAKE_CXX_ABI:-}" ]]; then
  CMAKE_CXX_ABI="$("${PYTHON_ABS}" -c "import torch; print(1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0)")"
fi
echo "CMAKE_CXX_ABI=${CMAKE_CXX_ABI} (must match PyTorch ABI)"

if [[ "${FORCE_CLEAN:-0}" == "1" ]]; then
  rm -rf "${BUILD}"
fi
mkdir -p "${BUILD}"
cd "${BUILD}"

if [[ ! -f CMakeCache.txt ]] || [[ "${FORCE_RECONFIGURE:-0}" == "1" ]]; then
  cmake "${DP}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL}" \
    -DPython_EXECUTABLE="${PYTHON_ABS}" \
    -DCMAKE_CXX_ABI="${CMAKE_CXX_ABI}"
else
  echo "Reusing existing CMake cache (set FORCE_RECONFIGURE=1 to reconfigure)."
fi

echo "Building (parallel jobs=${JOBS}) ..."
cmake --build . --parallel "${JOBS}"

echo "Installing to ${INSTALL} ..."
cmake --install .

# NumPy 2.x removed ``np.string_``; upstream PlaceDB still uses it at runtime.
PLACE_DB="${INSTALL}/dreamplace/PlaceDB.py"
if [[ -f "${PLACE_DB}" ]] && grep -q 'np\.string_' "${PLACE_DB}"; then
  echo "Patching PlaceDB.py for NumPy 2 (np.string_ -> np.bytes_) ..."
  sed -i 's/dtype=np\.string_/dtype=np.bytes_/g' "${PLACE_DB}"
fi

echo "Done. Smoke test:"
cd "${INSTALL}"
"${PYTHON_ABS}" dreamplace/Placer.py --help >/dev/null
echo "OK — run from install dir, e.g.: cd ${INSTALL} && python dreamplace/Placer.py test/ispd2005/adaptec1.json"
