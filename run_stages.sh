#!/usr/bin/env bash
# Preflight-gated pipeline runner.
#
# Identical to run_pipeline.sh but runs 00_Scripts/_preflight.py before each stage
# and refuses to launch one that fails the check. Written as a separate file so it
# could be created while run_pipeline.sh was mid-execution (bash reads scripts
# lazily; editing a running script corrupts it).
#
#   ./run_stages.sh 10a 10b            run the named stages
#   ./run_stages.sh                    run every stage listed below
#   SKIP_PREFLIGHT=1 ./run_stages.sh   bypass the gate (not advised)

set -uo pipefail
ROOT="/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC"
ENVN="floodcast_env"
cd "$ROOT" || exit 1
mkdir -p 99_Logs
TIMINGS="$ROOT/99_Logs/stage_timings.csv"
[ -f "$TIMINGS" ] || echo 'stage,script,date,start,end,duration_s,duration_human,status,source,notes' > "$TIMINGS"

hms() {
  local s=$1
  if [ "$s" -ge 3600 ]; then printf '%dh %02dm %02ds' $((s/3600)) $((s%3600/60)) $((s%60))
  else printf '%dm %02ds' $((s/60)) $((s%60)); fi
}
record() {
  printf '%s,%s,%s,%s,%s,%s,%s,%s,runner,""\n' \
    "$1" "$2" "$(date +%F)" "$3" "$4" "$5" "$(hms "$5")" "$6" >> "$TIMINGS"
}

run_stage() {                       # run_stage <tag> <script> [sentinel]
  local tag="$1" script="$2" sentinel="${3:-}"
  if [ -n "$sentinel" ] && [ -e "$sentinel" ]; then
    echo "[$(date +%H:%M:%S)] SKIP  $tag  (already done)"
    record "$tag" "$script" "-" "-" 0 SKIPPED
    return 0
  fi

  if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
    echo "[$(date +%H:%M:%S)] CHECK $tag"
    if ! conda run --no-capture-output -n "$ENVN" python3 00_Scripts/_preflight.py \
            "$script" > "99_Logs/${tag}.preflight.log" 2>&1; then
      echo "[$(date +%H:%M:%S)] BLOCK $tag  - preflight failed, NOT launching"
      cat "99_Logs/${tag}.preflight.log"
      record "$tag" "$script" "-" "-" 0 "BLOCKED(preflight)"
      exit 3
    fi
    grep -E "^  (syntax|undefined|imports|input dir|per-size)" "99_Logs/${tag}.preflight.log" | sed 's/^/        /'
  fi

  local st=$(date +%H:%M:%S) t0=$SECONDS
  echo "[$st] START $tag"
  PYTHONUNBUFFERED=1 conda run --no-capture-output -n "$ENVN" \
      python3 "00_Scripts/$script" > "99_Logs/${tag}.log" 2>&1
  local rc=$? el=$((SECONDS - t0)) en=$(date +%H:%M:%S)
  if [ $rc -ne 0 ]; then
    echo "[$en] FAIL  $tag  (exit $rc, $(hms $el))"
    record "$tag" "$script" "$st" "$en" "$el" "FAIL(exit $rc)"
    echo "--- last 20 lines ---"; tail -20 "99_Logs/${tag}.log"
    exit $rc
  fi
  echo "[$en] OK    $tag  ($(hms $el))"
  record "$tag" "$script" "$st" "$en" "$el" OK
}

declare -a WANT=("$@")
want() {
  [ ${#WANT[@]} -eq 0 ] && return 0
  local s; for s in "${WANT[@]}"; do [[ "$1" == "$s"* ]] && return 0; done
  return 1
}

echo "==================================================================="
echo "FSM pipeline (preflight-gated)   $(date '+%Y-%m-%d %H:%M:%S')"
echo "==================================================================="

want 10a && run_stage 10a_figures            10a_figures_maps.py
want 10b && run_stage 10b_maps_all_sizes     10b_generate_maps_all_sizes.py
want 10c && run_stage 10c_fix_comparison     10c_fix_comparison_maps.py
want 10d && run_stage 10d_manuscript         10d_setup_manuscript_folder.py

want 11a && run_stage 11a_codebook           11a_export_codebook.py
want 11b && run_stage 11b_raba_lulc          11b_export_raba_lulc_conversion.py
want 11c && run_stage 11c_scripts_summary    11c_export_scripts_summary.py
want 11d && run_stage 11d_summary_report     11d_generate_summary_report.py

echo "==================================================================="
echo "PIPELINE COMPLETE  $(date '+%Y-%m-%d %H:%M:%S')"
echo "==================================================================="
