#!/bin/bash
# Build a current OpenROAD + Yosys into the ORFS tree, using ORFS's own
# build_openroad.sh in --local (non-Docker) mode. No sudo required:
# common deps go to $HOME/.local via DependencyInstaller.sh -common -local.
#
# After this finishes, point evaluate_with_orfs.py at the freshly built
# binaries via:
#   OPENROAD_EXE=/u5/w223zhan/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad
#   YOSYS_EXE=/u5/w223zhan/OpenROAD-flow-scripts/tools/install/yosys/bin/yosys
#
# Submit:
#   sbatch scripts/slurm/build_openroad.slurm
#
#SBATCH --job-name=build_openroad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=sweep_logs/slurm_%j_%x.out
#SBATCH --error=sweep_logs/slurm_%j_%x.err

set -euo pipefail

ORFS_ROOT="${ORFS_ROOT:-/u5/w223zhan/OpenROAD-flow-scripts}"
cd "${ORFS_ROOT}"

echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-local}  Start: $(date)"
echo "ORFS_ROOT=${ORFS_ROOT}"

JOBS="${SLURM_CPUS_PER_TASK:-16}"
BUILD_LOG="${ORFS_ROOT}/tools/OpenROAD/build/openroad_build.log"
INSTALL_OPENROAD="${ORFS_ROOT}/tools/install/OpenROAD/bin/openroad"
INSTALL_YOSYS="${ORFS_ROOT}/tools/install/yosys/bin/yosys"

# OR-Tools (from DependencyInstaller -local) ships protobuf 31.x headers.  If
# /opt/anaconda3 is on PATH, its protobuf 6.x headers win and par/gpl/mpl fail
# with "Protobuf C++ gencode is built with an incompatible version".
_sanitize_openroad_build_env() {
  # Drop conda/anaconda from PATH so pip/cmake/g++ do not pick them up.
  local _path=""
  local _part
  IFS=':' read -ra _parts <<< "${PATH:-}"
  for _part in "${_parts[@]}"; do
    [[ -z "${_part}" ]] && continue
    if [[ "${_part}" == *anaconda* || "${_part}" == *conda/envs* ]]; then
      continue
    fi
    _path="${_path:+${_path}:}${_part}"
  done
  export PATH="${ORFS_ROOT}/dependencies/bin:${HOME}/.local/bin:${_path}"
  export LD_LIBRARY_PATH="${HOME}/.local/lib:${HOME}/.local/lib64:${ORFS_ROOT}/dependencies/lib:${ORFS_ROOT}/dependencies/lib64:${LD_LIBRARY_PATH:-}"

  unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PYTHON_EXE || true
  unset CPLUS_INCLUDE_PATH C_INCLUDE_PATH CPATH PKG_CONFIG_PATH || true
  unset CMAKE_PREFIX_PATH PYTHONPATH CFLAGS CXXFLAGS || true
  export CMAKE_IGNORE_PREFIX_PATH="/opt/anaconda3;/opt/anaconda3/lib/cmake"
}

_has_or_tools() {
  [[ -f "${HOME}/.local/or-tools/lib/libortools.so" \
    || -f "${HOME}/.local/lib/libortools.so" \
    || -f "${HOME}/.local/lib64/libortools.so" ]] \
    && [[ -d "${HOME}/.local/or-tools/include/google/protobuf" ]]
}

_openroad_cmake_cache_polluted() {
  local _cache="${ORFS_ROOT}/tools/OpenROAD/build/CMakeCache.txt"
  [[ -f "${_cache}" ]] && grep -q '/opt/anaconda3' "${_cache}"
}

_ensure_yaml_cpp_local() {
  if [[ -f "${HOME}/.local/lib/cmake/yaml-cpp/yaml-cpp-config.cmake" ]]; then
    return 0
  fi
  local _src="${ORFS_ROOT}/tools/kepler-formal/thirdparty/yaml-cpp"
  local _b="${ORFS_ROOT}/tools/.yaml-cpp-build"
  if [[ ! -f "${_src}/CMakeLists.txt" ]]; then
    echo "ERROR: vendored yaml-cpp missing at ${_src}" >&2
    return 1
  fi
  echo "==[yaml-cpp] install to ${HOME}/.local (anaconda yaml-cpp pulls /opt/anaconda3/include) =="
  _sanitize_openroad_build_env
  cmake -S "${_src}" -B "${_b}" \
    -DCMAKE_INSTALL_PREFIX="${HOME}/.local" \
    -DCMAKE_BUILD_TYPE=Release \
    -DYAML_BUILD_SHARED_LIBS=ON
  cmake --build "${_b}" -j "${JOBS}"
  cmake --install "${_b}"
}

_openroad_cmake_flags() {
  # ABSL in ~/.local/include disagrees with or-tools libs (TestCFileUtils link fail).
  # openroad_deps_prefixes.txt sets ABSL_ROOT=~/.local; override to or-tools.
  cat <<EOF
-DBUILD_GUI=OFF
-DENABLE_TESTS=OFF
-DCMAKE_IGNORE_PREFIX_PATH=/opt/anaconda3
-DProtobuf_DIR=${HOME}/.local/or-tools/lib/cmake/protobuf
-Dortools_DIR=${HOME}/.local/or-tools/lib/cmake/ortools
-Dabsl_DIR=${HOME}/.local/or-tools/lib/cmake/absl
-DABSL_ROOT=${HOME}/.local/or-tools
-DBoost_DIR=${HOME}/.local/lib/cmake/Boost-1.89.0
-DBoost_ROOT=${HOME}/.local
-DTBB_DIR=${HOME}/.local/lib/cmake/TBB
-DCapnProto_DIR=${HOME}/.local/lib/cmake/CapnProto
-Dyaml-cpp_DIR=${HOME}/.local/lib/cmake/yaml-cpp
-DPython3_EXECUTABLE=/usr/bin/python3
-DCMAKE_INSTALL_PREFIX=${ORFS_ROOT}/tools/install/OpenROAD
-DCMAKE_INSTALL_RPATH=${ORFS_ROOT}/dependencies/lib:${ORFS_ROOT}/dependencies/lib64:${HOME}/.local/or-tools/lib:${HOME}/.local/lib
-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE
EOF
}

_sanitize_openroad_build_env

# Stage 1: OR-Tools into $HOME/.local (skip if already installed).
if _has_or_tools; then
  echo "==[1/2] or-tools present in ${HOME}/.local; skipping DependencyInstaller =="
  echo "       (avoids pip -U upgrading anaconda protobuf and breaking the C++ build)"
else
  echo "==[1/2] etc/DependencyInstaller.sh -common -local ====================="
  _sanitize_openroad_build_env
  mkdir -p "${SLURM_SUBMIT_DIR:-${ORFS_ROOT}}/sweep_logs"
  _dep_log="${SLURM_SUBMIT_DIR:-${ORFS_ROOT}}/sweep_logs/dep_installer_${SLURM_JOB_ID:-local}.log"
  bash etc/DependencyInstaller.sh -common -local 2>&1 | tee "${_dep_log}" | tail -200
  _sanitize_openroad_build_env
fi

# yaml-cpp from anaconda exposes YAML_CPP_INCLUDE_DIR=/opt/anaconda3/include,
# which drags protobuf 6.x ahead of or-tools protobuf 31.x during par compile.
_ensure_yaml_cpp_local

# Stage 2: build Yosys + OpenROAD locally.
echo
echo "==[2/2] build_openroad.sh --local -t ${JOBS} ========================="
_sanitize_openroad_build_env
export CMAKE_PREFIX_PATH="${ORFS_ROOT}/dependencies:${HOME}/.local:${HOME}/.local/or-tools"

echo "Removing prior OpenROAD build dir for clean configure"
rm -rf "${ORFS_ROOT}/tools/OpenROAD/build"
echo "Removing prior kepler-formal build dir for clean TBB configure"
rm -rf "${ORFS_ROOT}/tools/kepler-formal/build"

_openroad_args="$(_openroad_cmake_flags | tr '\n' ' ')"
echo "OpenROAD cmake flags: ${_openroad_args}"

mkdir -p "$(dirname "${BUILD_LOG}")"
set +e
bash build_openroad.sh --local --threads "${JOBS}" --openroad-args "${_openroad_args}" 2>&1 | tee "${BUILD_LOG}"
_build_rc=${PIPESTATUS[0]}
set -e

if [[ "${_build_rc}" -ne 0 ]]; then
  echo
  echo "==[build failed: last compiler errors] ================================="
  grep -E 'error:|Error [0-9]|fatal error|PROTOBUF_VERSION|incompatible version' "${BUILD_LOG}" 2>/dev/null | tail -40 || true
  exit "${_build_rc}"
fi

echo
echo "==[summary] =========================================================="
ls -la "${INSTALL_OPENROAD}" "${INSTALL_YOSYS}" 2>&1 || true
"${INSTALL_OPENROAD}" -version 2>&1 | head -3 || true
"${INSTALL_YOSYS}" -V 2>&1 | awk '{print $1, $2}' | head -1 || true

echo "Done: $(date)"
