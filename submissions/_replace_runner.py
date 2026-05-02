"""Process wrapper for the bundled RePlAce binary."""

from __future__ import annotations

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


@dataclass(frozen=True)
class ReplaceConfig:
    """One RePlAce invocation configuration."""

    density: float = 0.80
    pcofmax: float = 1.03
    extra_args: Sequence[str] = ()

    def args(self) -> List[str]:
        if not (0.0 < float(self.density) <= 1.0):
            raise ValueError(f"density must be in (0, 1], got {self.density!r}")
        if float(self.pcofmax) <= 0.0:
            raise ValueError(f"pcofmax must be positive, got {self.pcofmax!r}")
        return [
            "-den",
            _fmt_float(self.density),
            "-pcofmax",
            _fmt_float(self.pcofmax),
            *[str(v) for v in self.extra_args],
        ]


@dataclass(frozen=True)
class ReplaceRunResult:
    """Result of one RePlAce process run."""

    config: ReplaceConfig
    returncode: int
    timed_out: bool
    runtime_seconds: float
    cwd: Path
    log_path: Path
    output_dir: Path
    pl_paths: List[Path]

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and bool(self.pl_paths)

    @property
    def usable(self) -> bool:
        """Whether this run produced placement files worth true-proxy screening."""

        return bool(self.pl_paths) and (
            self.returncode == 0 or self.timed_out or self.returncode < 0
        )


def run_replace(
    export: BookshelfExport,
    config: ReplaceConfig,
    *,
    binary_path: Path | str = Path(
        "external/MacroPlacement/Flows/util/RePlAceFlow/RePlAce-static"
    ),
    timeout_seconds: float = 600.0,
    log_dir: Path | str | None = None,
    stop_after_first_pl: bool = True,
) -> ReplaceRunResult:
    """Run RePlAce for one exported Bookshelf testcase.

    RePlAce's ``-bmflag etc`` mode expects this layout:

    ``<cwd>/ETC/<bookshelf_name>/<bookshelf_name>.aux``

    The exporter can write anywhere, but this runner intentionally requires the
    above layout so failures happen before launching a long process.
    """

    cwd = _replace_cwd(export)
    binary = Path(binary_path)
    if not binary.is_absolute():
        binary = (Path.cwd() / binary).resolve()
    if not binary.exists():
        raise FileNotFoundError(binary)

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    log_root = Path(log_dir) if log_dir is not None else cwd / "replace_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / _log_name(export.bookshelf_name, config)
    raw_log_path = log_path.with_name(log_path.name + ".raw")

    before = set(_discover_experiment_dirs(cwd, export.bookshelf_name))
    cmd = [
        str(binary),
        "-bmflag",
        "etc",
        "-bmname",
        export.bookshelf_name,
        *config.args(),
    ]

    start = time.monotonic()
    timed_out = False
    early_stopped = False
    returncode = 0
    with raw_log_path.open("w", encoding="utf-8", errors="replace") as raw_log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            text=True,
            stdout=raw_log,
            stderr=subprocess.DEVNULL,
        )
        deadline = start + float(timeout_seconds)
        first_pl_time: float | None = None
        while proc.poll() is None:
            if stop_after_first_pl:
                if _fresh_pl_ready(cwd, export.bookshelf_name, before):
                    if first_pl_time is None:
                        first_pl_time = time.monotonic()
                    elif time.monotonic() - first_pl_time >= 0.03:
                        early_stopped = True
                        proc.kill()
                        break
                else:
                    first_pl_time = None
            if time.monotonic() >= deadline:
                timed_out = True
                proc.kill()
                break
            time.sleep(0.02)
        proc.wait()

    output = _read_log(raw_log_path)
    if timed_out:
        output += f"\n[TIMEOUT after {timeout_seconds} seconds]\n"
    returncode = 124 if timed_out else int(proc.returncode)
    runtime = time.monotonic() - start

    after = _discover_experiment_dirs(cwd, export.bookshelf_name)
    new_dirs = [p for p in after if p not in before]
    pl_paths = _discover_pl_paths(new_dirs, export.bookshelf_name)
    output_dir = cwd / "outputs" / "ETC" / export.bookshelf_name

    log_path.write_text(
        _log_header(cmd, cwd, runtime, returncode, timed_out, early_stopped) + output
    )
    raw_log_path.unlink(missing_ok=True)

    return ReplaceRunResult(
        config=config,
        returncode=returncode,
        timed_out=timed_out,
        runtime_seconds=runtime,
        cwd=cwd,
        log_path=log_path,
        output_dir=output_dir,
        pl_paths=pl_paths,
    )


def discover_replace_pls(export: BookshelfExport) -> List[Path]:
    """Return all known RePlAce placement outputs for ``export``."""

    cwd = _replace_cwd(export)
    dirs = _discover_experiment_dirs(cwd, export.bookshelf_name)
    return _discover_pl_paths(dirs, export.bookshelf_name)


def _replace_cwd(export: BookshelfExport) -> Path:
    directory = export.directory.resolve()
    if directory.name != export.bookshelf_name:
        raise ValueError(
            f"export directory must end with {export.bookshelf_name!r}: {directory}"
        )
    etc_dir = directory.parent
    if etc_dir.name != "ETC":
        raise ValueError(f"export directory must be under an ETC directory: {directory}")
    aux = directory / f"{export.bookshelf_name}.aux"
    if not aux.exists():
        raise FileNotFoundError(aux)
    return etc_dir.parent


def _discover_experiment_dirs(cwd: Path, bookshelf_name: str) -> List[Path]:
    root = cwd / "outputs" / "ETC" / bookshelf_name
    if not root.exists():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("experiment")]
    return sorted(dirs, key=lambda p: (_experiment_number(p), p.name))


def _discover_pl_paths(experiment_dirs: Sequence[Path], bookshelf_name: str) -> List[Path]:
    found: List[Path] = []
    for exp_dir in experiment_dirs:
        preferred = exp_dir / f"{bookshelf_name}.eplace-mGP2D.pl"
        if preferred.exists():
            found.append(preferred)
            continue
        found.extend(sorted(exp_dir.glob("*.pl")))
    return _dedupe_paths(found)


def _fresh_pl_ready(cwd: Path, bookshelf_name: str, before: set[Path]) -> bool:
    for pl_path in _discover_pl_paths(
        [p for p in _discover_experiment_dirs(cwd, bookshelf_name) if p not in before],
        bookshelf_name,
    ):
        if _pl_ready(pl_path):
            return True
    return False


def _pl_ready(path: Path) -> bool:
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size <= 0:
        return False
    try:
        with path.open("rb") as f:
            if size > 256:
                f.seek(-256, 2)
            tail = f.read()
    except OSError:
        return False
    return b"\n" in tail


def _dedupe_paths(paths: Sequence[Path]) -> List[Path]:
    seen = set()
    out: List[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(path)
    return out


def _experiment_number(path: Path) -> int:
    suffix = path.name.removeprefix("experiment")
    return int(suffix) if suffix.isdigit() else 10**9


def _fmt_float(value: float) -> str:
    return f"{float(value):.6g}"


def _log_name(bookshelf_name: str, config: ReplaceConfig) -> str:
    den = _fmt_float(config.density).replace(".", "p")
    pcof = _fmt_float(config.pcofmax).replace(".", "p")
    extra = "_".join(
        str(arg).strip("-").replace(".", "p").replace("/", "_")
        for arg in config.extra_args
    )
    suffix = f"_{extra}" if extra else ""
    return f"{bookshelf_name}_den{den}_pcof{pcof}{suffix}.log"


def _log_header(
    cmd: Sequence[str],
    cwd: Path,
    runtime: float,
    returncode: int,
    timed_out: bool,
    early_stopped: bool,
) -> str:
    return (
        f"cmd: {' '.join(cmd)}\n"
        f"cwd: {cwd}\n"
        f"runtime_seconds: {runtime:.3f}\n"
        f"returncode: {returncode}\n"
        f"timed_out: {timed_out}\n"
        f"early_stopped: {early_stopped}\n"
        "\n"
    )


def _process_output(stdout: str | None, stderr: str | None) -> str:
    chunks = []
    if stdout:
        chunks.append(stdout)
    if stderr:
        chunks.append("\n[stderr]\n")
        chunks.append(stderr)
    return "".join(chunks)


def _read_log(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _timeout_output(exc: subprocess.TimeoutExpired) -> str:
    chunks = []
    if exc.stdout:
        chunks.append(exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout)
    if exc.stderr:
        chunks.append(exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr)
    chunks.append(f"\n[TIMEOUT after {exc.timeout} seconds]\n")
    return "".join(chunks)
