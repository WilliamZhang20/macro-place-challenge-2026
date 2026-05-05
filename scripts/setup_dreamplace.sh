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
#   PARSER_GDSII=ON|OFF (default: ON)
#     DREAMPlace links optional Limbo GDSII support into some modules (e.g.
#     draw_place). Disabling it without also compiling out those codepaths
#     yields missing symbols at import time.
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

# Default ON: required for a working import of modules that reference GdsParser.
: "${PARSER_GDSII:=ON}"

# DREAMPlace's CMake build (NVCC) spawns intermediate CUDA tools like `cudafe++`.
# When `nvcc` is found via an absolute path (e.g., from a conda "targets" directory),
# those intermediate tools may still be resolved via `PATH`.
# Prepending the interpreter's directory makes `cudafe++` (and friends) discoverable.
PYTHON_DIR="$(dirname "${PYTHON_ABS}")"
export PATH="${PYTHON_DIR}:${PATH}"

CONDA_PREFIX="$(dirname "${PYTHON_DIR}")"
# DREAMPlace's pybind11 extensions link against `libstdc++.so.6`. If PyTorch is
# imported first, the dynamic loader may bind the *system* libstdc++ (older
# CXXABI) and later fail loading DREAMPlace ops with:
#   version `CXXABI_1.3.15' not found
# Prepending the env's lib dir fixes resolution order for this process tree
# (build, install, smoke test, and any child processes that inherit env).
if [[ -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  # See ``_dreamplace_cpu_smoke._ctypes_preload`` rationale: some CUDA wheels
  # use RUNPATH that pulls an older system libstdc++ unless we preload the env's.
  if [[ -f "${CONDA_PREFIX}/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
  fi
fi
# Make system Boost/Flex headers visible to conda toolchains that only search
# `${CONDA_PREFIX}/include` and `${CONDA_PREFIX}/targets/.../include`.
if [[ -d "${CONDA_PREFIX}/include" ]]; then
  if [[ -d /usr/include/boost && ! -e "${CONDA_PREFIX}/include/boost" ]]; then
    ln -s /usr/include/boost "${CONDA_PREFIX}/include/boost"
  fi
  if [[ -f /usr/include/FlexLexer.h && ! -e "${CONDA_PREFIX}/include/FlexLexer.h" ]]; then
    ln -s /usr/include/FlexLexer.h "${CONDA_PREFIX}/include/FlexLexer.h"
  fi
fi
export CXXFLAGS="${CXXFLAGS:-} -DBOOST_MATH_DISABLE_FLOAT128 -UBOOST_MATH_USE_FLOAT128"
# NVCC invokes device tools (e.g., `cicc`) that live under the conda env's NVVM
# component directory rather than directly in `${CONDA_PREFIX}/bin`.
NVVM_BIN="${CONDA_PREFIX}/nvvm/bin"
if [[ -d "${NVVM_BIN}" ]]; then
  export PATH="${NVVM_BIN}:${PATH}"
fi

# CUDA include dirs (for device-link/fatbin compilation stages)
CUDA_TOOLCHAIN_INCLUDE_DIR="${CONDA_PREFIX}/targets/x86_64-linux/include"
if [[ -d "${CUDA_TOOLCHAIN_INCLUDE_DIR}" ]]; then
  # Preserve user-provided CMAKE_CUDA_FLAGS (often includes -gencode),
  # but ensure we always pass toolchain include paths so headers like
  # `fatbinary_section.h` can be found during device linking.
  CMAKE_CUDA_FLAGS="${CMAKE_CUDA_FLAGS:-} -I${CUDA_TOOLCHAIN_INCLUDE_DIR}"

  # On Debian/Ubuntu, glibc splits some internal headers under the
  # multiarch directory (e.g. `/usr/include/x86_64-linux-gnu/bits`).
  # Conda GCC + NVCC sometimes ends up parsing `/usr/include/features-time64.h`
  # without the multiarch include path, which breaks with:
  #   fatal error: bits/timesize.h: No such file or directory
  # Adding this as a *system* include keeps it late in the search order
  # and avoids the older `bits/` symlink hacks that can break libstdc++.
  if [[ -d /usr/include/x86_64-linux-gnu ]]; then
    CMAKE_CUDA_FLAGS="${CMAKE_CUDA_FLAGS} -isystem /usr/include/x86_64-linux-gnu"
  fi

  # NVCC's device-link sometimes compiles a temporary
  # `/tmp/..._intermediate_link.fatbin.c` that includes
  # `"fatbinary_section.h"` without propagating our `-I` flags.
  # As a last-resort safety net, place a copy in `/tmp` so the
  # compiler can always find it via the source directory.
  if [[ -f "${CUDA_TOOLCHAIN_INCLUDE_DIR}/fatbinary_section.h" ]]; then
    cp -f "${CUDA_TOOLCHAIN_INCLUDE_DIR}/fatbinary_section.h" /tmp/fatbinary_section.h || true
  fi
fi

# Some conda "cuda-nvcc" toolchains expect CRT stub files under
# `${CONDA_PREFIX}/targets/<arch>-linux/bin/crt/`, but the stubs may only
# be shipped under `${CONDA_PREFIX}/bin/crt/`. DREAMPlace's CUDA device link
# fails hard if those stubs are missing, so we symlink them when needed.
TARGETS_CRT_DIR="${CONDA_PREFIX}/targets/x86_64-linux/bin/crt"
if [[ ! -d "${TARGETS_CRT_DIR}" && -d "${CONDA_PREFIX}/bin/crt" ]]; then
  mkdir -p "${TARGETS_CRT_DIR}"
fi
if [[ -d "${TARGETS_CRT_DIR}" ]]; then
  # Patch `crt/link.stub` include style:
  # - upstream conda toolchains sometimes use `#include <fatbinary_section.h>`,
  #   which relies on NVCC/host compiler "system include" paths.
  # - we mirror the header into `.../bin/crt/`, so switch to
  #   `#include "fatbinary_section.h"` to prefer the local `crt/` directory.
  if [[ -f "${CONDA_PREFIX}/bin/crt/link.stub" ]]; then
    LINK_STUB_SRC="${CONDA_PREFIX}/bin/crt/link.stub"
    LINK_STUB_DST="${TARGETS_CRT_DIR}/link.stub"
    rm -f "${LINK_STUB_DST}" || true
    sed 's@#include <fatbinary_section.h>@#include "fatbinary_section.h"@' "${LINK_STUB_SRC}" > "${LINK_STUB_DST}"
  fi
  if [[ -f "${CONDA_PREFIX}/bin/crt/prelink.stub" && ! -f "${TARGETS_CRT_DIR}/prelink.stub" ]]; then
    ln -sf "${CONDA_PREFIX}/bin/crt/prelink.stub" "${TARGETS_CRT_DIR}/prelink.stub"
  fi

  # Also mirror the CRT headers expected under `crt/<header>.h` includes.
  INCLUDE_CRT_DIR="${CONDA_PREFIX}/targets/x86_64-linux/include/crt"
  if [[ -d "${INCLUDE_CRT_DIR}" ]]; then
    for item in "${INCLUDE_CRT_DIR}"/*; do
      base="$(basename "${item}")"
      if [[ ! -e "${TARGETS_CRT_DIR}/${base}" ]]; then
        ln -s "${item}" "${TARGETS_CRT_DIR}/${base}"
      fi
    done

    # Some toolchain setups add `${TARGETS_CRT_DIR}` itself to the include path,
    # which makes `#include "crt/host_defines.h"` resolve to:
    #   ${TARGETS_CRT_DIR}/crt/host_defines.h
    NESTED_CRT_DIR="${TARGETS_CRT_DIR}/crt"
    mkdir -p "${NESTED_CRT_DIR}"
    for item in "${INCLUDE_CRT_DIR}"/*; do
      base="$(basename "${item}")"
      if [[ ! -e "${NESTED_CRT_DIR}/${base}" ]]; then
        ln -s "${item}" "${NESTED_CRT_DIR}/${base}"
      fi
    done

    # `crt/link.stub` also includes `fatbinary_section.h` (not under `crt/`),
    # so provide it in the expected lookup locations too.
    TARGETS_INCLUDE_DIR="${CONDA_PREFIX}/targets/x86_64-linux/include"
    if [[ -f "${TARGETS_INCLUDE_DIR}/fatbinary_section.h" ]]; then
      for dst in "${TARGETS_CRT_DIR}/fatbinary_section.h" "${NESTED_CRT_DIR}/fatbinary_section.h"; do
        if [[ ! -e "${dst}" ]]; then
          ln -s "${TARGETS_INCLUDE_DIR}/fatbinary_section.h" "${dst}"
        fi
      done
    fi
  fi
fi

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
  # Help CMake/FindCUDA in environments where CUDA driver libraries are
  # installed system-wide but not inside the conda toolchain. In particular,
  # `CUDA_CUDA_LIBRARY` may otherwise be left as NOTFOUND, which breaks
  # linking of certain CUDA-aware targets (e.g. timing_heterosta_cpp).
  CUDA_CUDA_LIBRARY_DEFAULT=""
  if [[ -z "${CUDA_CUDA_LIBRARY:-}" ]]; then
    if [[ -f /usr/lib/x86_64-linux-gnu/libcuda.so ]]; then
      CUDA_CUDA_LIBRARY_DEFAULT="/usr/lib/x86_64-linux-gnu/libcuda.so"
    fi
  fi

  BOOST_INCLUDE_DIR_DEFAULT=""
  BOOST_LIBRARY_DIR_DEFAULT=""
  BOOST_GRAPH_LIBRARY_DEFAULT=""
  BOOST_REGEX_LIBRARY_DEFAULT=""
  if [[ -d /usr/include/boost ]]; then
    BOOST_INCLUDE_DIR_DEFAULT="/usr/include"
  fi
  if [[ -d /usr/lib/x86_64-linux-gnu ]]; then
    BOOST_LIBRARY_DIR_DEFAULT="/usr/lib/x86_64-linux-gnu"
    if [[ -f /usr/lib/x86_64-linux-gnu/libboost_graph.so ]]; then
      BOOST_GRAPH_LIBRARY_DEFAULT="/usr/lib/x86_64-linux-gnu/libboost_graph.so"
    fi
    if [[ -f /usr/lib/x86_64-linux-gnu/libboost_regex.so ]]; then
      BOOST_REGEX_LIBRARY_DEFAULT="/usr/lib/x86_64-linux-gnu/libboost_regex.so"
    fi
  fi

  CMAKE_EXTRA_ARGS=()
  if [[ -n "${CUDA_CUDA_LIBRARY_DEFAULT}" ]]; then
    # If a previous configure left `CUDA_CUDA_LIBRARY` at NOTFOUND in CMakeCache,
    # CMake may keep the bad cache entry unless we explicitly unset it first.
    CMAKE_EXTRA_ARGS+=(-UCUDA_CUDA_LIBRARY "-DCUDA_CUDA_LIBRARY=${CUDA_CUDA_LIBRARY_DEFAULT}")
  fi

  # NVCC uses its own "host compiler" (`-ccbin`) which may otherwise pick up the
  # conda cross GCC. That combination frequently breaks when NVCC parses the
  # distro glibc headers under `/usr/include` (see `bits/floatn*.h` errors).
  # Default to the system `g++` when available; users can override via env/cmake.
  CUDA_HOST_COMPILER_DEFAULT=""
  if [[ -z "${CUDA_HOST_COMPILER:-}" ]]; then
    if [[ -x /usr/bin/g++ ]]; then
      CUDA_HOST_COMPILER_DEFAULT="/usr/bin/g++"
    fi
  fi
  if [[ -n "${CUDA_HOST_COMPILER_DEFAULT}" ]]; then
    CMAKE_EXTRA_ARGS+=(-UCUDA_HOST_COMPILER "-DCUDA_HOST_COMPILER=${CUDA_HOST_COMPILER_DEFAULT}")
  fi

  # Limbo splits GDSII into stream/ascii components. Stale CMake cache entries can
  # leave `PARSER_GDSII=ON` while `PARSER_GDSII_STREAM` stays OFF, which skips
  # building `libgdsparser.a` and breaks Python imports (undefined GdsParser symbols).
  if [[ "${PARSER_GDSII}" != "OFF" ]]; then
    CMAKE_EXTRA_ARGS+=(
      -UPARSER_GDSII_STREAM
      -DPARSER_GDSII_STREAM=ON
    )
  fi

  cmake "${DP}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL}" \
    -DPython_EXECUTABLE="${PYTHON_ABS}" \
    -DCMAKE_CXX_ABI="${CMAKE_CXX_ABI}" \
    -DPARSER_GDSII="${PARSER_GDSII}" \
    -DCMAKE_CUDA_FLAGS="${CMAKE_CUDA_FLAGS:-}" \
    "${CMAKE_EXTRA_ARGS[@]}" \
    ${BOOST_INCLUDE_DIR_DEFAULT:+-DBoost_INCLUDE_DIR="${BOOST_INCLUDE_DIR_DEFAULT}"} \
    ${BOOST_LIBRARY_DIR_DEFAULT:+-DBoost_LIBRARY_DIR_RELEASE="${BOOST_LIBRARY_DIR_DEFAULT}"} \
    ${BOOST_GRAPH_LIBRARY_DEFAULT:+-DBoost_GRAPH_LIBRARY_RELEASE="${BOOST_GRAPH_LIBRARY_DEFAULT}"} \
    ${BOOST_REGEX_LIBRARY_DEFAULT:+-DBoost_REGEX_LIBRARY_RELEASE="${BOOST_REGEX_LIBRARY_DEFAULT}"}
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
env LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" "${PYTHON_ABS}" dreamplace/Placer.py --help >/dev/null
echo "OK — run from install dir, e.g.: cd ${INSTALL} && python dreamplace/Placer.py test/ispd2005/adaptec1.json"
echo "Tip: if imports fail with CXXABI errors, activate conda and/or export:"
echo '  LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"'
echo '  LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"'
