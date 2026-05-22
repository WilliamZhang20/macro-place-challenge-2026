#!/bin/bash
# Run the DREAMPlace pipeline submission via the Docker-like environment
# described in eval_docker/Dockerfile.dreamplace_pipeline, using udocker
# (rootless, daemonless — cluster docker daemon restricted to docker
# group, which we are not in).
#
# CRITICAL: The host's external/DREAMPlace/install/*.so was built against
# host's torch 2.10.0+cu128 — ABI-incompatible with the container's
# torch 2.5.1.  So we replicate the Dockerfile's RUN steps inside the
# container to (1) install Limbo build deps, (2) install CMake 3.28,
# (3) cmake-build DREAMPlace from source so the .so files match the
# container's torch.  Build is one-time per container; subsequent
# `udocker run` calls reuse the same writable rootfs.
#
#SBATCH --job-name=dreamplace_udocker
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --output=sweep_logs/slurm_%j_%x.out
#SBATCH --error=sweep_logs/slurm_%j_%x.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"
mkdir -p sweep_logs

STAMP="$(date +%Y%m%d_%H%M%S)"
BUILD_LOG="sweep_logs/dreamplace_udocker_build_${SLURM_JOB_ID:-local}_${STAMP}.log"

echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-}  Start: $(date)"
echo "Repo: ${REPO_ROOT}"
echo "Build log: ${BUILD_LOG}"

UDOCKER="${UDOCKER:-${HOME}/.local/bin/udocker}"
"${UDOCKER}" --version | head -1

# Use the -devel image so we get nvcc + CUDA dev headers needed to build
# DREAMPlace from source.  Matches the Dockerfile's PYTORCH_BASE arg.
#
# CRITICAL: DREAMPlace pin_pos_cuda_segment_kernel.cu is incompatible with
# CUDA 12.4's CUB (BlockLoadAlgorithm moved to cub::detail::, breaking build
# at ~53%).  We pin to CUDA 12.8 / torch 2.10 to match the host env exactly,
# guaranteeing identical build + proxy behavior inside vs outside the container.
BASE_IMAGE="pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel"
CTR_NAME="dreamplace_udocker_${SLURM_JOB_ID:-local}"

if ! "${UDOCKER}" images 2>/dev/null | grep -q "${BASE_IMAGE}"; then
  echo "[udocker] pulling ${BASE_IMAGE} ..."
  "${UDOCKER}" pull "${BASE_IMAGE}"
fi

"${UDOCKER}" rm "${CTR_NAME}" 2>/dev/null || true
"${UDOCKER}" create --name="${CTR_NAME}" "${BASE_IMAGE}"
"${UDOCKER}" setup --nvidia "${CTR_NAME}" 2>&1 | tail -3 || true

# Step 0: probe the container's Python layout (newer pytorch images dropped
# /opt/conda and ship a PEP-668-marked system python3, so pip + cmake must
# target the actual interpreter rather than hard-coded /opt/conda paths).
echo ""
echo "[udocker] [0/4] probing container python layout ..."
# udocker prepends "executing: nvidia_entrypoint.sh" + a CUDA banner to every
# run's stdout, so we wrap the probe payload in PYBIN= markers and grep for
# the marker to extract only our actual path.
PYTHON_PROBE_RAW="$( "${UDOCKER}" run --user=root "${CTR_NAME}" /bin/bash -c '
for cand in /opt/conda/bin/python /opt/conda/bin/python3 /usr/local/bin/python3 /usr/bin/python3; do
  if [ -x "$cand" ]; then echo "PYBIN=$cand"; exit 0; fi
done
WHICHED="$(command -v python3 || command -v python || true)"
[ -n "$WHICHED" ] && echo "PYBIN=$WHICHED"
' 2>/dev/null )"
PYTHON_PROBE="$(echo "${PYTHON_PROBE_RAW}" | grep -oE 'PYBIN=[^[:space:]]+' | head -1 | sed 's/^PYBIN=//')"
if [ -z "${PYTHON_PROBE}" ]; then
  echo "[udocker] [0/4] FATAL: could not locate a python interpreter inside ${BASE_IMAGE}"
  "${UDOCKER}" rm "${CTR_NAME}" 2>/dev/null || true
  exit 1
fi
PYTHON_BIN="${PYTHON_PROBE}"
PYTHON_LIB_DIR="$(dirname "$(dirname "${PYTHON_BIN}")")/lib"
echo "[udocker] [0/4] PYTHON_BIN=${PYTHON_BIN}  PYTHON_LIB_DIR=${PYTHON_LIB_DIR}"

# Step 1: apt deps for Limbo + Cairo (root) -----------------------------------
echo ""
echo "[udocker] [1/4] installing apt build deps as root ..."
"${UDOCKER}" run --user=root "${CTR_NAME}" /bin/bash -c "
set -e
apt-get update -qq && \
apt-get install -y --no-install-recommends \
  build-essential cmake flex bison libfl-dev \
  libboost-dev libboost-graph-dev libboost-iostreams-dev libboost-system-dev \
  libboost-program-options-dev libboost-thread-dev libboost-regex-dev \
  libboost-filesystem-dev libboost-serialization-dev libboost-chrono-dev \
  libcairo2-dev libgmp-dev zlib1g-dev pkg-config wget curl ca-certificates \
  tcl && \
rm -rf /var/lib/apt/lists/* && \
# pytorch:2.10 image ships /usr/local/bin/cmake (4.x) which shadows the apt
# cmake (3.28.x).  cmake 4.x dropped CMP0048 OLD that DREAMPlace's bundled
# lemon depends on, so wipe the shadow binaries.
rm -f /usr/local/bin/cmake /usr/local/bin/ctest /usr/local/bin/cpack \
      /usr/local/bin/cmake-gui && \
echo '--- which cmake ---' && which cmake && cmake --version
" >>"${BUILD_LOG}" 2>&1
echo "[udocker] [1/4] apt deps OK (see ${BUILD_LOG})"

# Step 2: install pip deps + CMake 3.28 inside container ---------------------
echo ""
echo "[udocker] [2/4] installing pip deps + CMake 3.28 ..."
"${UDOCKER}" run --user=root "${CTR_NAME}" /bin/bash -c "
set -e
# --break-system-packages: required for PEP-668 system pythons (e.g. PyTorch
# 2.10 image's /usr/bin/python3.12). Harmless on conda-based images.
# cmake comes from apt (Ubuntu 24.04 ships 3.28.x); we explicitly DO NOT pip-
# install cmake because the pip wheel is cmake 4.x which removed CMP0048 OLD
# that DREAMPlace's bundled Limbo/lemon still relies on.
${PYTHON_BIN} -m pip install --break-system-packages --no-cache-dir \
  'pyunpack>=0.1.2' 'patool>=1.12' 'matplotlib>=3.3' 'cairocffi>=0.9.0' \
  'pkgconfig>=1.4.0' 'setuptools>=39.1.0' 'scipy>=1.10' 'numpy>=1.24,<2.3' \
  'shapely>=2.0' 'torch_optimizer==0.3.0' 'ncg_optimizer==0.2.2' \
  'tqdm>=4.65' 'absl-py>=1.0' && \
echo '--- which cmake ---' && which cmake && cmake --version
" >>"${BUILD_LOG}" 2>&1
echo "[udocker] [2/4] pip + cmake OK"

# Step 3: build DREAMPlace inside container (matches Dockerfile RUN cmake)
#
# CRITICAL: do NOT pipe to `| tail` — that mask cmake's exit code AND truncates
# the real error.  Pipe full output to BUILD_LOG; on failure we tail-dump the
# log and exit so the failure is visible in the slurm .out.
echo ""
echo "[udocker] [3/4] building DREAMPlace from source (~20-30 min) ..."
if ! "${UDOCKER}" run --user=root \
  --volume="${REPO_ROOT}:/challenge" \
  --workdir=/challenge \
  "${CTR_NAME}" /bin/bash -c "
set -e
rm -rf /challenge/external/DREAMPlace/build /challenge/external/DREAMPlace/install
cmake -S /challenge/external/DREAMPlace -B /challenge/external/DREAMPlace/build \
  -DCMAKE_INSTALL_PREFIX=/challenge/external/DREAMPlace/install \
  -DPython_EXECUTABLE=${PYTHON_BIN} \
  -DCMAKE_CXX_ABI=1 \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DPARSER_GDSII=ON \
  -DPARSER_GDSII_STREAM=ON \
  -DCMAKE_CUDA_FLAGS='-gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_90,code=sm_90'
cmake --build /challenge/external/DREAMPlace/build --parallel 8
cmake --install /challenge/external/DREAMPlace/build
test -f /challenge/external/DREAMPlace/install/dreamplace/Placer.py
sed -i 's/dtype=np\.string_/dtype=np.bytes_/g' /challenge/external/DREAMPlace/install/dreamplace/PlaceDB.py 2>/dev/null || true
echo 'DREAMPlace build OK: Placer.py present'
" >>"${BUILD_LOG}" 2>&1; then
  echo ""
  echo "[udocker] [3/4] BUILD FAILED -- last 200 lines of ${BUILD_LOG}:"
  echo "--------------------------------------------------------------------------------"
  tail -n 200 "${BUILD_LOG}"
  echo "--------------------------------------------------------------------------------"
  "${UDOCKER}" rm "${CTR_NAME}" 2>/dev/null || true
  exit 1
fi
echo "[udocker] [3/4] DREAMPlace build OK"

# Step 3.5: sanity check the new install loads ------------------------------
echo ""
echo "[udocker] [3.5/4] sanity check DREAMPlace subprocess ..."
if ! "${UDOCKER}" run \
  --hostauth \
  --user=$(id -u):$(id -g) \
  --workdir=/challenge/external/DREAMPlace/install \
  --volume="${REPO_ROOT}:/challenge" \
  --env=LD_LIBRARY_PATH=${PYTHON_LIB_DIR} \
  "${CTR_NAME}" /bin/bash -c "${PYTHON_BIN} dreamplace/Placer.py --help" >>"${BUILD_LOG}" 2>&1; then
  echo "[udocker] [3.5/4] SMOKE TEST FAILED -- last 100 lines of ${BUILD_LOG}:"
  echo "--------------------------------------------------------------------------------"
  tail -n 100 "${BUILD_LOG}"
  echo "--------------------------------------------------------------------------------"
  "${UDOCKER}" rm "${CTR_NAME}" 2>/dev/null || true
  exit 1
fi
echo "[udocker] [3.5/4] smoke test OK"

# Step 4: run the evaluator --------------------------------------------------
EVAL_ARGS="${EVAL_ARGS:---all}"
LOG_FILE="sweep_logs/dreamplace_udocker_${SLURM_JOB_ID:-local}_${STAMP}.log"

echo ""
echo "[udocker] [4/4] starting evaluator at $(date)"
echo "[udocker] args: ${EVAL_ARGS}"
echo "[udocker] log: ${LOG_FILE}"

"${UDOCKER}" run \
  --hostauth \
  --user=$(id -u):$(id -g) \
  --workdir=/challenge \
  --volume="${REPO_ROOT}:/challenge" \
  --env=PYTHONPATH=/challenge:/challenge/submissions \
  --env=MACRO_PLACE_DP_GPU=auto \
  --env=MACRO_PLACE_TUNER_DEBUG=1 \
  --env=PYTHONUNBUFFERED=1 \
  --env=LD_LIBRARY_PATH=${PYTHON_LIB_DIR} \
  "${CTR_NAME}" \
  ${PYTHON_BIN} -m macro_place.evaluate /challenge/submissions/dreamplace_pipeline_placer.py ${EVAL_ARGS} 2>&1 | tee "${LOG_FILE}"

RC=${PIPESTATUS[0]}

"${UDOCKER}" rm "${CTR_NAME}" 2>/dev/null || true

echo "Done: $(date) rc=${RC}"
exit ${RC}
