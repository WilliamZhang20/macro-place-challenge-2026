#!/usr/bin/env bash
set -u

# Overnight IBM sweep for all placer variants.
#
# Runs each placer/benchmark pair independently with a hard timeout so slow
# cases do not block the rest of the sweep. Logs are written under
# sweep_logs/<timestamp>/.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1200}"
LOG_ROOT="${LOG_ROOT:-sweep_logs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$LOG_ROOT/$STAMP"
SUMMARY="$OUT_DIR/summary.csv"

IBM_BENCHMARKS=(
  ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10
  ibm11 ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18
)

DEFAULT_PLACERS=(
  submissions/casadi_placer.py
  submissions/dccp_placer.py
  submissions/hard_coord_descent_placer.py
  submissions/hard_macro_lns_quick_placer.py
)

mkdir -p "$OUT_DIR"

if [[ -f "$HOME/myenv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/myenv/bin/activate"
elif command -v uv >/dev/null 2>&1; then
  EVALUATE_CMD=(uv run evaluate)
else
  echo "ERROR: neither ~/myenv/bin/activate nor uv is available." >&2
  exit 1
fi

if ! declare -p EVALUATE_CMD >/dev/null 2>&1; then
  EVALUATE_CMD=(evaluate)
fi

if [[ "$#" -gt 0 ]]; then
  PLACERS=("$@")
else
  PLACERS=()
  for placer in "${DEFAULT_PLACERS[@]}"; do
    if [[ -f "$placer" ]]; then
      PLACERS+=("$placer")
    fi
  done
fi

{
  echo "timestamp,placer,benchmark,status,exit_code,runtime_seconds,proxy,wirelength,density,congestion,overlaps,log"
} > "$SUMMARY"

echo "Writing logs to $OUT_DIR"
echo "Timeout per placer/benchmark: ${TIMEOUT_SECONDS}s"
echo "Placers:"
printf '  %s\n' "${PLACERS[@]}"
echo

run_one() {
  local placer="$1"
  local bench="$2"
  local safe_placer
  local log
  local start
  local end
  local runtime
  local status
  local exit_code
  local line
  local proxy
  local wl
  local den
  local cong
  local overlaps

  safe_placer="${placer//\//__}"
  safe_placer="${safe_placer%.py}"
  log="$OUT_DIR/${safe_placer}__${bench}.log"

  start="$(date +%s)"
  timeout "$TIMEOUT_SECONDS" "${EVALUATE_CMD[@]}" "$placer" -b "$bench" > "$log" 2>&1
  exit_code="$?"
  end="$(date +%s)"
  runtime="$((end - start))"

  if [[ "$exit_code" -eq 0 ]]; then
    status="ok"
  elif [[ "$exit_code" -eq 124 ]]; then
    status="timeout"
  else
    status="fail"
  fi

  line="$(grep -E 'proxy=[0-9]' "$log" | tail -n 1 || true)"
  proxy=""
  wl=""
  den=""
  cong=""
  overlaps=""
  if [[ -n "$line" ]]; then
    proxy="$(sed -n 's/.*proxy=\([0-9.][0-9.]*\).*/\1/p' <<< "$line")"
    wl="$(sed -n 's/.*wl=\([0-9.][0-9.]*\).*/\1/p' <<< "$line")"
    den="$(sed -n 's/.*den=\([0-9.][0-9.]*\).*/\1/p' <<< "$line")"
    cong="$(sed -n 's/.*cong=\([0-9.][0-9.]*\).*/\1/p' <<< "$line")"
    if grep -q 'VALID' <<< "$line"; then
      overlaps="0"
    else
      overlaps="$(sed -n 's/.*INVALID (\([0-9][0-9]*\) overlaps).*/\1/p' <<< "$line")"
    fi
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$(date -Iseconds)" "$placer" "$bench" "$status" "$exit_code" "$runtime" \
    "$proxy" "$wl" "$den" "$cong" "$overlaps" "$log" >> "$SUMMARY"

  printf '%-45s %-6s %-8s %5ss proxy=%s\n' "$placer" "$bench" "$status" "$runtime" "${proxy:-NA}"
}

for placer in "${PLACERS[@]}"; do
  if [[ ! -f "$placer" ]]; then
    echo "Skipping missing placer: $placer" >&2
    continue
  fi
  for bench in "${IBM_BENCHMARKS[@]}"; do
    run_one "$placer" "$bench"
  done
done

echo
echo "Done. Summary: $SUMMARY"
