"""DREAMPlace wiring: export Bookshelf, subprocess Placer, import .gp.pl.

``gpu`` in the JSON is resolved for CPU- or CUDA-built installs (see
``resolve_dreamplace_gpu``). Requires a built tree under ``external/DREAMPlace/install``
(``scripts/setup_dreamplace.sh``).

By default Placer stdout/stderr are discarded so long logs cannot fill pipe buffers
and deadlock the parent (``capture_output``). Set ``MACRO_PLACE_DP_DEBUG_SUBPROCESS=1``
to inherit the parent terminal and see DREAMPlace logs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _prepend_ld_library_path(dir_path: Path) -> None:
    """Ensure *dir_path* is searched first by the dynamic linker (Linux).

    PyTorch may load a system ``libstdc++.so.6`` first; DREAMPlace extensions
    built with a newer conda toolchain then fail with ``CXXABI_1.3.15`` unless
    the env's ``libstdc++`` wins the lookup order.
    """

    if sys.platform != "linux" or not dir_path.is_dir():
        return
    key = "LD_LIBRARY_PATH"
    prefix = str(dir_path)
    cur = os.environ.get(key, "")
    if cur == "":
        os.environ[key] = prefix
        return
    if cur == prefix or cur.startswith(prefix + os.pathsep):
        return
    os.environ[key] = prefix + os.pathsep + cur


def _ensure_toolchain_libstdcxx_preload() -> None:
    """Best-effort: put the active Python env's lib dir ahead of system libs."""

    conda = os.environ.get("CONDA_PREFIX", "").strip()
    if conda:
        lib_dir = Path(conda) / "lib"
        _prepend_ld_library_path(lib_dir)
        _ctypes_preload(lib_dir / "libstdc++.so.6")
        return
    # venv layout: ``.../env/bin/python`` -> ``.../env/lib``
    lib_dir = Path(sys.executable).resolve().parent.parent / "lib"
    _prepend_ld_library_path(lib_dir)
    _ctypes_preload(lib_dir / "libstdc++.so.6")


def _ctypes_preload(so_path: Path) -> None:
    """``LD_LIBRARY_PATH`` alone is not always enough.

    NVIDIA's pip wheels (e.g. cuDNN) ship shared libs with a ``RUNPATH`` that
    can steer *transitive* ``libstdc++.so.6`` resolution away from the conda
    env, even when ``LD_LIBRARY_PATH`` prefixes that env. Loading the env's
    ``libstdc++`` explicitly first avoids ``CXXABI_1.3.15`` mismatch failures
    when importing PyTorch before DREAMPlace extension modules.
    """

    if sys.platform != "linux" or not so_path.is_file():
        return
    try:
        import ctypes
    except ImportError:
        return
    try:
        ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return


_ensure_toolchain_libstdcxx_preload()

import json
import re
import subprocess
import tempfile
from typing import Any, Dict, Mapping, Optional

import torch

from macro_place.benchmark import Benchmark

from _replace_bookshelf import write_bookshelf
from _replace_import import import_bookshelf_placement
from _hard_legalizer import legalize_hard

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_INSTALL = _REPO_ROOT / "external" / "DREAMPlace" / "install"


def _tuner_progress_enabled() -> bool:
    return os.environ.get("MACRO_PLACE_TUNER_DEBUG", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
_PLACER_REL = Path("dreamplace") / "Placer.py"
_CONFIGURE_REL = Path("dreamplace") / "configure.py"


def default_dreamplace_install() -> Path:
    return _DEFAULT_INSTALL


def dreamplace_install_ok(root: Path | None = None) -> tuple[bool, str]:
    """Return whether ``install/`` looks runnable."""
    inst = Path(root) if root is not None else _DEFAULT_INSTALL
    placer = inst / _PLACER_REL
    cfg = inst / _CONFIGURE_REL
    if not placer.is_file():
        return False, f"missing {placer}"
    if not cfg.is_file():
        return False, f"missing {cfg}"
    return True, "ok"


def dreamplace_built_with_cuda(install: Path | str) -> bool:
    """True if this DREAMPlace install was compiled with CUDA (see ``configure.py``)."""

    cfg_path = Path(install) / _CONFIGURE_REL
    if not cfg_path.is_file():
        return False
    text = cfg_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r'"CUDA_FOUND"\s*:\s*"([^"]*)"', text)
    if not m:
        return False
    return m.group(1).strip().upper() == "TRUE"


def read_use_gpu_env() -> Optional[bool]:
    """Parse ``MACRO_PLACE_DP_GPU``: ``0|cpu|off`` → False, ``1|gpu|on`` → True, ``auto`` → None."""

    raw = os.environ.get("MACRO_PLACE_DP_GPU", "").strip().lower()
    if raw in ("0", "false", "no", "off", "cpu"):
        return False
    if raw in ("1", "true", "yes", "on", "gpu"):
        return True
    if raw in ("auto", ""):
        return None
    return None


def resolve_dreamplace_gpu(
    install: Path | str,
    *,
    use_gpu: Optional[bool],
    dreamplace_json_overrides: Optional[Mapping[str, Any]],
    torch_cuda_available: Optional[bool] = None,
) -> int:
    """Return ``0`` or ``1`` for DREAMPlace JSON ``gpu`` (never asserts on CPU-only builds).

    Resolution order:

    1. ``MACRO_PLACE_DP_GPU`` env if set.
    2. ``use_gpu`` argument if not ``None`` (``True`` / ``False``).
    3. If ``dreamplace_json_overrides`` contains ``gpu``, coerce to bool then clamp.
    4. Otherwise **auto**: ``1`` iff the install has CUDA **and** PyTorch sees a GPU.

    If the requested device is GPU but the install is CPU-only or PyTorch has no CUDA,
    returns ``0``.
    """

    if torch_cuda_available is None:
        torch_cuda_available = bool(torch.cuda.is_available())

    dp_cuda = dreamplace_built_with_cuda(install)
    can_use_gpu = dp_cuda and torch_cuda_available

    pref = read_use_gpu_env()
    if pref is None:
        pref = use_gpu
    if pref is None and dreamplace_json_overrides and "gpu" in dreamplace_json_overrides:
        try:
            pref = bool(int(dreamplace_json_overrides["gpu"]))
        except (TypeError, ValueError):
            pref = bool(dreamplace_json_overrides["gpu"])
    if pref is None:
        pref = can_use_gpu

    want = bool(pref) and can_use_gpu
    return 1 if want else 0


def deep_merge_dreamplace_json(
    base: Dict[str, Any], overrides: Mapping[str, Any]
) -> Dict[str, Any]:
    """Recursively merge ``overrides`` into ``base`` (dict branches merge; scalars replace).

    Lists are merged **only** when both sides are lists of dicts (e.g. ``global_place_stages``):
    each index is deep-merged so tuning can patch ``learning_rate`` / ``optimizer`` in stage 0
    without respecifying bins or iteration counts.
    """

    out = dict(base)
    for key, val in overrides.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(val, Mapping)
        ):
            out[key] = deep_merge_dreamplace_json(
                dict(out[key]), val  # type: ignore[arg-type]
            )
        elif (
            key in out
            and isinstance(out[key], list)
            and isinstance(val, list)
            and out[key]
            and val
            and all(isinstance(x, dict) for x in out[key])
            and all(isinstance(x, Mapping) for x in val)
        ):
            merged: list[Any] = []
            n_merge = min(len(out[key]), len(val))
            for i in range(len(out[key])):
                if i < n_merge:
                    merged.append(
                        deep_merge_dreamplace_json(dict(out[key][i]), val[i])
                    )
                else:
                    merged.append(out[key][i])
            out[key] = merged
        else:
            out[key] = val
    return out


def _dp_json(
    *,
    aux_abs: Path,
    result_dir_abs: Path,
    global_iterations: int,
    num_bins: int,
    num_threads: int,
    target_density: float,
) -> Dict[str, Any]:
    # ``gpu`` finalized in ``run_dreamplace_placement`` via ``resolve_dreamplace_gpu``.
    return {
        "aux_input": str(aux_abs.resolve()),
        "gpu": 0,
        "num_bins_x": num_bins,
        "num_bins_y": num_bins,
        "global_place_stages": [
            {
                "num_bins_x": num_bins,
                "num_bins_y": num_bins,
                "iteration": int(global_iterations),
                "learning_rate": 0.01,
                "wirelength": "weighted_average",
                "optimizer": "nesterov",
                "Llambda_density_weight_iteration": 1,
                "Lsub_iteration": 1,
            }
        ],
        "target_density": float(target_density),
        "density_weight": 8e-5,
        "gamma": 4.0,
        "random_seed": 1000,
        "scale_factor": 1.0,
        "ignore_net_degree": 100,
        "enable_fillers": 1,
        "gp_noise_ratio": 0.025,
        "global_place_flag": 1,
        "legalize_flag": 1,
        # Bookshelf exports mix row-snapped fillers with non-row-height macros; abacus
        # legalization asserts (node_size_y vs row_height). Greedy legalization suffices.
        "abacus_legalize_flag": 0,
        "detailed_place_flag": 0,
        "detailed_place_engine": "",
        "detailed_place_command": "",
        "stop_overflow": 0.12,
        "dtype": "float32",
        "plot_flag": 0,
        "random_center_init_flag": 0,
        "gift_init_flag": 0,
        "sort_nets_by_degree": 0,
        "num_threads": int(num_threads),
        "deterministic_flag": 1,
        "timing_opt_flag": 0,
        "result_dir": str(result_dir_abs.resolve()),
    }


def run_dreamplace_placement(
    benchmark: Benchmark,
    plc,
    *,
    dreamplace_install: Path | None = None,
    global_iterations: int = 20,
    num_bins: int = 128,
    num_threads: int = 4,
    target_density: float = 0.72,
    timeout_seconds: float = 900.0,
    subprocess_env: Optional[Dict[str, str]] = None,
    dreamplace_json_overrides: Optional[Mapping[str, Any]] = None,
    use_gpu: Optional[bool] = None,
    initial_placement: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Export Bookshelf, run DREAMPlace once, import + legalize. Or ``None`` if failed.

    ``dreamplace_json_overrides`` is deep-merged onto the default JSON so you can tune
    any key DREAMPlace accepts (``target_density``, ``density_weight``, extra
    ``global_place_stages``, ``macro_place_flag``, etc.). Paths ``aux_input`` and
    ``result_dir`` are always set after merging so the run stays consistent with the
    exported Bookshelf.

    ``use_gpu``: ``None`` = auto (GPU if the DREAMPlace build and PyTorch both support
    CUDA), ``False`` = CPU, ``True`` = request GPU when capable. Override with env
    ``MACRO_PLACE_DP_GPU=0|1|auto``. A merged ``gpu`` key in overrides participates
    when ``use_gpu`` and env are unset (see ``resolve_dreamplace_gpu``).

    ``initial_placement``: optional full macro tensor for Bookshelf seed (defaults to
    ``benchmark.macro_positions``). Use for multi-start diversity (e.g. jittered
    handoff).
    """

    inst = Path(dreamplace_install) if dreamplace_install is not None else _DEFAULT_INSTALL
    ok, reason = dreamplace_install_ok(inst)
    if not ok:
        return None

    with tempfile.TemporaryDirectory(prefix="dp_cpu_smoke_") as tmp:
        tmp_path = Path(tmp)
        etc = tmp_path / "ETC"
        # DREAMPlace's Bookshelf shapes parser rejects our minimal ``UCLA shapes`` stub;
        # export without .shapes (RePlAce bridge uses the same workaround).
        # Omit .route as well: DREAMPlace's Bookshelf route parser rejects our grid header;
        # CPU smoke runs global place without this contest-specific routing overlay.
        seed = (
            initial_placement.float()
            if initial_placement is not None
            else benchmark.macro_positions.clone().float()
        )
        export = write_bookshelf(
            benchmark,
            plc,
            etc,
            bookshelf_name=benchmark.name,
            scale=1000,
            include_route=False,
            include_shapes=False,
            soft_macro_mode="row_height",
            initial_placement=seed,
        )
        result_root = tmp_path / "results"
        result_root.mkdir(parents=True, exist_ok=True)
        cfg_path = tmp_path / "dp_smoke.json"
        cfg: Dict[str, Any] = _dp_json(
            aux_abs=export.aux_path,
            result_dir_abs=result_root,
            global_iterations=global_iterations,
            num_bins=num_bins,
            num_threads=num_threads,
            target_density=target_density,
        )
        if dreamplace_json_overrides:
            cfg = deep_merge_dreamplace_json(cfg, dict(dreamplace_json_overrides))
        cfg["aux_input"] = str(export.aux_path.resolve())
        cfg["result_dir"] = str(result_root.resolve())
        cfg["gpu"] = resolve_dreamplace_gpu(
            inst,
            use_gpu=use_gpu,
            dreamplace_json_overrides=dreamplace_json_overrides,
        )

        effective_threads = int(cfg.get("num_threads", num_threads))
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")

        env = dict(os.environ)
        if subprocess_env:
            env.update(subprocess_env)
        env.setdefault("OMP_NUM_THREADS", str(effective_threads))

        cmd = [sys.executable, str(inst / _PLACER_REL), str(cfg_path)]
        if _tuner_progress_enabled():
            print(
                f"[tune:dp] {benchmark.name}  spawning Placer  gpu={cfg.get('gpu')}  "
                f"iters={global_iterations}  bins={num_bins}  "
                f"timeout={timeout_seconds:.0f}s",
                file=sys.stderr,
                flush=True,
            )
        # Never use capture_output=True here: DREAMPlace logs can exceed the pipe
        # buffer; the child then blocks on write while we block in communicate().
        debug_io = os.environ.get("MACRO_PLACE_DP_DEBUG_SUBPROCESS", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        run_kw: Dict[str, Any] = {
            "cwd": str(inst),
            "env": env,
            "timeout": timeout_seconds,
            "stdin": subprocess.DEVNULL,
        }
        if debug_io:
            proc = subprocess.run(cmd, **run_kw)
        else:
            proc = subprocess.run(
                cmd,
                **run_kw,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        if proc.returncode != 0:
            return None

        design = export.bookshelf_name
        gp_pl = result_root / design / f"{design}.gp.pl"
        if not gp_pl.is_file():
            return None

        try:
            raw = import_bookshelf_placement(
                gp_pl,
                export.metadata_path,
                benchmark,
                clamp_to_canvas=True,
                keep_fixed=True,
            )
        except Exception:
            return None

        return legalize_hard(raw, benchmark, legalize_rounds=1200, overlap_gap=1e-3)


# Backwards-compatible name
run_dreamplace_cpu_placement = run_dreamplace_placement

