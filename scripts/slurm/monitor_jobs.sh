#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: monitor_jobs.sh [--interval SECONDS] [--log-dir DIR] [--out-dir DIR] JOB_ID...

Monitors Slurm jobs with squeue/scontrol and records compact state snapshots.
Log contents are not read during normal RUNNING/PENDING polling. To enable live
keyword scanning, create $OUT_DIR/enable_log_scan. Logs are always scanned once
when a job is no longer visible in squeue.
USAGE
}

interval=60
log_dir="sweep_logs"
out_dir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval)
      interval="${2:?missing interval}"
      shift 2
      ;;
    --log-dir)
      log_dir="${2:?missing log dir}"
      shift 2
      ;;
    --out-dir)
      out_dir="${2:?missing out dir}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

job_ids=("$@")
job_csv="$(IFS=,; echo "${job_ids[*]}")"

if [[ -z "$out_dir" ]]; then
  out_dir="${log_dir}/monitor_${job_csv//,/_}"
fi

mkdir -p "$out_dir"

status_log="$out_dir/status.tsv"
events_log="$out_dir/events.log"
alerts_log="$out_dir/alerts.log"
files_log="$out_dir/log_files.tsv"
control_file="$out_dir/enable_log_scan"

touch "$status_log" "$events_log" "$alerts_log" "$files_log"

if [[ ! -s "$status_log" ]]; then
  printf 'timestamp\tjob_id\tstate\ttime\ttime_limit\tnodes\treason_or_nodelist\n' >> "$status_log"
fi
if [[ ! -s "$files_log" ]]; then
  printf 'timestamp\tjob_id\tbytes\tpath\n' >> "$files_log"
fi

declare -A missing_seen=()
declare -A scanned_after_exit=()

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

log_event() {
  printf '%s %s\n' "$(timestamp)" "$*" >> "$events_log"
}

job_files() {
  local job_id="$1"
  find "$log_dir" -maxdepth 1 -type f -name "*${job_id}*" -print 2>/dev/null | sort
}

record_log_file_sizes() {
  local now="$1"
  local job_id="$2"
  local file size
  while IFS= read -r file; do
    [[ -n "$file" ]] || continue
    size="$(stat -c '%s' "$file" 2>/dev/null || printf '0')"
    printf '%s\t%s\t%s\t%s\n' "$now" "$job_id" "$size" "$file" >> "$files_log"
  done < <(job_files "$job_id")
}

scan_logs_for_alerts() {
  local job_id="$1"
  local reason="$2"
  local now file matches
  now="$(timestamp)"

  while IFS= read -r file; do
    [[ -n "$file" ]] || continue
    # Keep this bounded: only the recent tail is inspected, and only matching lines are saved.
    matches="$(
      tail -n 300 "$file" 2>/dev/null |
        grep -Ein 'error|failed|failure|traceback|exception|segmentation fault|cannot|no such file|killed|out of memory|oom|abort|fatal' |
        tail -n 80 || true
    )"
    if [[ -n "$matches" ]]; then
      {
        printf '%s job=%s reason=%s file=%s\n' "$now" "$job_id" "$reason" "$file"
        printf '%s\n' "$matches"
        printf '\n'
      } >> "$alerts_log"
    fi
  done < <(job_files "$job_id")
}

all_done() {
  local job_id
  for job_id in "${job_ids[@]}"; do
    [[ "${missing_seen[$job_id]:-0}" == "1" ]] || return 1
  done
  return 0
}

log_event "monitor started jobs=${job_csv} interval=${interval}s log_dir=${log_dir} out_dir=${out_dir}"
log_event "live log scanning is disabled until ${control_file} exists; exit scanning remains enabled"

while true; do
  now="$(timestamp)"
  squeue_output="$(squeue -h -j "$job_csv" -o '%i|%T|%M|%l|%D|%R' 2>>"$events_log" || true)"

  for job_id in "${job_ids[@]}"; do
    line="$(printf '%s\n' "$squeue_output" | awk -F'|' -v id="$job_id" '$1 == id {print; exit}')"
    record_log_file_sizes "$now" "$job_id"

    if [[ -n "$line" ]]; then
      IFS='|' read -r _job_id state elapsed limit nodes reason <<< "$line"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$now" "$job_id" "$state" "$elapsed" "$limit" "$nodes" "$reason" >> "$status_log"

      if [[ -f "$control_file" ]]; then
        scan_logs_for_alerts "$job_id" "live_scan_${state}"
      fi
    else
      detail="$(scontrol show job "$job_id" 2>/dev/null | tr '\n' ' ' | sed -E 's/[[:space:]]+/ /g' || true)"
      [[ -n "$detail" ]] || detail="not visible in squeue or scontrol"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$now" "$job_id" "NOT_IN_SQUEUE" "-" "-" "-" "$detail" >> "$status_log"

      if [[ "${missing_seen[$job_id]:-0}" != "1" ]]; then
        missing_seen[$job_id]=1
        log_event "job ${job_id} left squeue; ${detail}"
      fi
      if [[ "${scanned_after_exit[$job_id]:-0}" != "1" ]]; then
        scanned_after_exit[$job_id]=1
        scan_logs_for_alerts "$job_id" "left_squeue"
      fi
    fi
  done

  if all_done; then
    log_event "all monitored jobs have left squeue; monitor exiting"
    exit 0
  fi

  sleep "$interval"
done
