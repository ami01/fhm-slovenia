#!/usr/bin/env bash
# Unattended run of the FSM Slovenia pipeline (LULC), stages 05c -> 11.
#
#   ./run_pipeline.sh              run every remaining stage
#   ./run_pipeline.sh 06 07        run only the named stages
#   SKIP_HEAVY=1 ./run_pipeline.sh skip full-Slovenia inference (stage 06)
#
# Each stage logs to 99_Logs/<stage>.log. The run stops at the first failure so a
# broken stage never feeds garbage downstream; re-invoke with the stage name to
# resume. Stages that already completed are skipped via their sentinel files.

set -uo pipefail
ROOT="/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC"
ENVN="floodcast_env"
cd "$ROOT" || exit 1
mkdir -p 99_Logs

TIMINGS="$ROOT/99_Logs/stage_timings.csv"
if [ ! -f "$TIMINGS" ]; then
  echo 'stage,script,date,start,end,duration_s,duration_human,status,source,notes' > "$TIMINGS"
fi

hms() {                             # seconds -> "12m 34s" / "2h 05m 03s"
  local s=$1
  if [ "$s" -ge 3600 ]; then printf '%dh %02dm %02ds' $((s/3600)) $((s%3600/60)) $((s%60))
  else printf '%dm %02ds' $((s/60)) $((s%60)); fi
}

record() {                          # record <tag> <script> <start> <end> <secs> <status>
  printf '%s,%s,%s,%s,%s,%s,%s,%s,runner,""\n' \
    "$1" "$2" "$(date +%F)" "$3" "$4" "$5" "$(hms "$5")" "$6" >> "$TIMINGS"
}

run_stage() {                       # run_stage <tag> <script> [sentinel]
  local tag="$1" script="$2" sentinel="${3:-}"
  if [ -n "$sentinel" ] && [ -e "$sentinel" ]; then
    echo "[$(date +%H:%M:%S)] SKIP  $tag  (already done: ${sentinel#$ROOT/})"
    record "$tag" "$script" "-" "-" 0 SKIPPED
    return 0
  fi
  local st=$(date +%H:%M:%S) t0=$SECONDS
  echo "[$st] START $tag"
  PYTHONUNBUFFERED=1 conda run --no-capture-output -n "$ENVN" \
      python3 "00_Scripts/$script" > "99_Logs/${tag}.log" 2>&1
  local rc=$? el=$((SECONDS - t0)) en=$(date +%H:%M:%S)
  if [ $rc -ne 0 ]; then
    echo "[$en] FAIL  $tag  (exit $rc, $(hms $el))"
    record "$tag" "$script" "$st" "$en" "$el" "FAIL(exit $rc)"
    echo "--- last 20 lines of 99_Logs/${tag}.log ---"
    tail -20 "99_Logs/${tag}.log"
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
echo "FSM Slovenia pipeline (LULC)   $(date '+%Y-%m-%d %H:%M:%S')"
echo "==================================================================="

# 05d — DNN seed variation (cheap, quantifies noise for the manuscript)
want 05d && run_stage 05d_dnn_seed_variation 05d_dnn_seed_variation.py \
    "$ROOT/05_Results/dnn_seed_variation.csv"

# 05c — train all sample sizes 1x,2x,3x,4x,10x (RF, XGB, CatBoost, DNN)
want 05c && run_stage 05c_train_multisample 05c_train_models_multisample.py \
    "$ROOT/05_Results/comparison/sample_size_comparison.csv"

# 06b — full-Slovenia probability maps for EVERY sample size  [VERY HEAVY]
# RF + XGBoost + CatBoost + DNN + ensemble mean, for 1x/2x/3x/4x/10x.
# 25 full-grid inference passes over 259M cells. Resumable: skips a size whose
# 5 maps already exist, so an interrupted run continues where it stopped.
if [ "${SKIP_HEAVY:-0}" != "1" ]; then
  want 06b && run_stage 06b_predict_all_sizes 06b_predict_probability_maps_all_sizes.py \
      "$ROOT/06_Probability_Maps/10x/FSM_Ensemble_mean_probability.tif"
else
  echo "[$(date +%H:%M:%S)] SKIP  06b (SKIP_HEAVY=1)"
fi

# 07 — threshold optimisation, classification, validation
want 07 && run_stage 07_threshold_classify_validate 07_threshold_classify_validate.py \
    "$ROOT/08_Validation/validation_metrics.csv"

# 08a — merge the DNN row into validation + comparison plots (needs 07)
want 08a && run_stage 08a_dnn_merge_validation 08a_dnn_merge_validation.py

# 08b-d — spatial CV, multiscale validation, SHAP
want 08b && run_stage 08b_spatial_cv 08b_spatial_cv.py
want 08c && run_stage 08c_multiscale 08c_validate_classify_spatial_cv_multiscale.py
want 08d && run_stage 08d_shap 08d_shap_analysis.py

# 09 — area, frequency ratio, sub-basin
want 09a && run_stage 09a_classify_area 09a_classify_area_analysis.py
want 09b && run_stage 09b_fr_area 09b_analysis_fr_area.py
want 09c && run_stage 09c_subbasin 09c_subbasin_analysis.py

# 10 — figures
want 10a && run_stage 10a_figures 10a_figures_maps.py
want 10b && run_stage 10b_maps_all_sizes 10b_generate_maps_all_sizes.py
want 10c && run_stage 10c_fix_comparison 10c_fix_comparison_maps.py
want 10d && run_stage 10d_manuscript 10d_setup_manuscript_folder.py

# 11 — documentation
want 11a && run_stage 11a_codebook 11a_export_codebook.py
want 11b && run_stage 11b_raba_lulc 11b_export_raba_lulc_conversion.py
want 11c && run_stage 11c_scripts_summary 11c_export_scripts_summary.py
want 11d && run_stage 11d_summary_report 11d_generate_summary_report.py

echo "==================================================================="
echo "PIPELINE COMPLETE  $(date '+%Y-%m-%d %H:%M:%S')"
echo "==================================================================="
