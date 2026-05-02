"""Process wrapper for DREAMPlace Bookshelf runs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

_SUBMISSIONS_DIR = Path(__file__).resolve().parent
if str(_SUBMISSIONS_DIR) not in sys.path:
    sys.path.insert(0, str(_SUBMISSIONS_DIR))

from _replace_bookshelf import BookshelfExport  # noqa: E402


def dreamplace_available(
    dreamplace_root: Path | str = Path("external/DREAMPlace"),
) -> tuple[bool, str]:
    """Return whether DREAMPlace appears configured enough to run."""

    root = _resolve_dreamplace_root(dreamplace_root)
    placer = root / "dreamplace" / "Placer.py"
    configure = root / "dreamplace" / "configure.py"
    if not placer.exists():
        return False, f"missing {placer}"
    if not configure.exists():
        return False, f"missing generated {configure}"
    return True, "ok"


@dataclass(frozen=True)
class DreamPlaceConfig:
    """One DREAMPlace invocation configuration."""

    target_density: float = 0.80
    num_bins_x: int = 0
    num_bins_y: int = 0
    iterations: int = 1000
    learning_rate: float = 0.01
    density_weight: float = 8e-5
    gpu: bool = False
    random_seed: int = 1000
    extra_params: dict | None = None


@dataclass(frozen=True)
class DreamPlaceRunResult:
    """Result of one DREAMPlace process run."""

    config: DreamPlaceConfig
    returncode: int
    timed_out: bool
    runtime_seconds: float
    cwd: Path
    params_path: Path
    log_path: Path
    result_dir: Path
    pl_paths: List[Path]

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and bool(self.pl_paths)

    @property
    def usable(self) -> bool:
        return bool(self.pl_paths) and (
            self.returncode == 0 or self.timed_out or self.returncode < 0
        )


def run_dreamplace(
    export: BookshelfExport,
    config: DreamPlaceConfig,
    *,
    dreamplace_root: Path | str = Path("external/DREAMPlace"),
    placer_path: Path | str | None = None,
    timeout_seconds: float = 600.0,
    work_root: Path | str | None = None,
) -> DreamPlaceRunResult:
    """Run DREAMPlace for one exported Bookshelf testcase."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    dreamplace_root = _resolve_dreamplace_root(dreamplace_root)
    placer = Path(placer_path) if placer_path is not None else dreamplace_root / "dreamplace" / "Placer.py"
    if not placer.is_absolute():
        placer = (Path.cwd() / placer).resolve()
    if not placer.exists():
        raise FileNotFoundError(placer)

    work = Path(work_root) if work_root is not None else export.directory.parent.parent / "dreamplace"
    work.mkdir(parents=True, exist_ok=True)
    result_dir = work / "results"
    params_path = work / _params_name(export.bookshelf_name, config)
    log_path = work / _log_name(export.bookshelf_name, config)
    params_path.write_text(
        json.dumps(_params_dict(export, config, result_dir), indent=2, sort_keys=True)
        + "\n"
    )

    cmd = [sys.executable, str(placer), str(params_path)]
    start = time.monotonic()
    timed_out = False
    try:
        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", str(work / "matplotlib"))
        _prepare_python_compat(work)
        env["PYTHONPATH"] = _prepend_pythonpath(
            work / "python_compat",
            env.get("PYTHONPATH"),
        )
        proc = subprocess.run(
            cmd,
            cwd=str(dreamplace_root),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=float(timeout_seconds),
            check=False,
        )
        output = proc.stdout or ""
        returncode = int(proc.returncode)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        output = _timeout_output(exc)
    runtime = time.monotonic() - start

    pl_paths = _collect_config_pls(
        discover_dreamplace_pls(export.bookshelf_name, result_dir),
        work,
        params_path.stem,
    )
    log_path.write_text(_log_header(cmd, dreamplace_root, runtime, returncode, timed_out) + output)

    return DreamPlaceRunResult(
        config=config,
        returncode=returncode,
        timed_out=timed_out,
        runtime_seconds=runtime,
        cwd=dreamplace_root,
        params_path=params_path,
        log_path=log_path,
        result_dir=result_dir,
        pl_paths=pl_paths,
    )


def discover_dreamplace_pls(bookshelf_name: str, result_dir: Path | str) -> List[Path]:
    result_dir = Path(result_dir)
    design_dir = result_dir / bookshelf_name
    preferred = design_dir / f"{bookshelf_name}.gp.pl"
    if preferred.exists():
        return [preferred]
    if not design_dir.exists():
        return []
    return sorted(design_dir.glob("*.pl"))


def _collect_config_pls(pl_paths: Sequence[Path], work: Path, run_stem: str) -> List[Path]:
    collected = work / "collected"
    collected.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []
    for index, pl_path in enumerate(pl_paths):
        dest = collected / f"{run_stem}_{index}_{pl_path.name}"
        shutil.copy2(pl_path, dest)
        out.append(dest)
    return out


def _prepare_python_compat(work: Path) -> None:
    compat_dir = work / "python_compat"
    compat_dir.mkdir(parents=True, exist_ok=True)
    (compat_dir / "sitecustomize.py").write_text(
        "import numpy as _np\n"
        "if not hasattr(_np, 'string_'):\n"
        "    _np.string_ = _np.bytes_\n"
    )


def _prepend_pythonpath(path: Path, existing: str | None) -> str:
    pieces = [str(path.resolve())]
    if existing:
        pieces.append(existing)
    return os.pathsep.join(pieces)


def _params_dict(
    export: BookshelfExport,
    config: DreamPlaceConfig,
    result_dir: Path,
) -> dict:
    params = {
        "aux_input": str(export.aux_path.resolve()),
        "gpu": 1 if config.gpu else 0,
        "num_bins_x": int(config.num_bins_x),
        "num_bins_y": int(config.num_bins_y),
        "global_place_stages": [
            {
                "num_bins_x": int(config.num_bins_x),
                "num_bins_y": int(config.num_bins_y),
                "iteration": int(config.iterations),
                "learning_rate": float(config.learning_rate),
                "wirelength": "weighted_average",
                "optimizer": "nesterov",
            }
        ],
        "target_density": float(config.target_density),
        "density_weight": float(config.density_weight),
        "random_seed": int(config.random_seed),
        "result_dir": str(result_dir.resolve()),
        "scale_factor": 1.0,
        "ignore_net_degree": 100,
        "enable_fillers": 1,
        "global_place_flag": 1,
        "legalize_flag": 1,
        "detailed_place_flag": 0,
        "stop_overflow": 0.07,
        "dtype": "float32",
        "num_threads": 8,
        "plot_flag": 0,
        "random_center_init_flag": 0,
    }
    if config.extra_params:
        params.update(config.extra_params)
    return params


def _params_name(bookshelf_name: str, config: DreamPlaceConfig) -> str:
    return (
        f"{bookshelf_name}_den{_fmt_float(config.target_density)}"
        f"_iter{int(config.iterations)}"
        f"_lr{_fmt_float(config.learning_rate)}"
        f"_dw{_fmt_float(config.density_weight)}_dreamplace.json"
    )


def _log_name(bookshelf_name: str, config: DreamPlaceConfig) -> str:
    return (
        f"{bookshelf_name}_den{_fmt_float(config.target_density)}"
        f"_iter{int(config.iterations)}"
        f"_lr{_fmt_float(config.learning_rate)}"
        f"_dw{_fmt_float(config.density_weight)}_dreamplace.log"
    )


def _fmt_float(value: float) -> str:
    return f"{float(value):.6g}".replace(".", "p")


def _timeout_output(exc: subprocess.TimeoutExpired) -> str:
    pieces = [f"[TIMEOUT after {exc.timeout} seconds]\n"]
    for attr in ("stdout", "stderr"):
        data = getattr(exc, attr, None)
        if not data:
            continue
        if isinstance(data, bytes):
            data = data.decode(errors="replace")
        pieces.append(str(data))
    return "".join(pieces)


def _log_header(
    cmd: Sequence[str],
    cwd: Path,
    runtime: float,
    returncode: int,
    timed_out: bool,
) -> str:
    return (
        f"$ {' '.join(cmd)}\n"
        f"cwd: {cwd}\n"
        f"runtime_seconds: {runtime:.3f}\n"
        f"returncode: {returncode}\n"
        f"timed_out: {timed_out}\n\n"
    )


def _resolve_dreamplace_root(dreamplace_root: Path | str) -> Path:
    """Prefer the installed DREAMPlace package when given the source root."""

    root = Path(dreamplace_root)
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    if (root / "dreamplace" / "configure.py").exists():
        return root
    install = root / "install"
    if (install / "dreamplace" / "configure.py").exists():
        return install.resolve()
    return root
