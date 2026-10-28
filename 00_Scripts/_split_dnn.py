"""Split train_dnn.py at the validation-merge boundary to break the 05b/06/07 cycle.

train_dnn.py welds two phases together:
  A) train the DNN and save model_dnn.pth   -- must run BEFORE stage 06
                                               (06 loads model_dnn.pth)
  B) merge the DNN row into validation_metrics.csv and redraw the 5-model
     comparison plots                        -- must run AFTER stage 07
                                               (07 creates validation_metrics.csv)

As one script it is unorderable. Split into 05b (phase A) and 08a (phase B), with
phase A persisting the four objects phase B needs.
"""
from pathlib import Path

DST = Path("/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/00_Scripts")
src = (DST / "05b_train_dnn.py").read_text().splitlines(keepends=True)

SPLIT = 383   # 1-indexed line of "# ── Merge with existing validation CSV ──"
assert "Merge with existing validation" in src[SPLIT - 1], src[SPLIT - 1]

head = "".join(src[:SPLIT - 1])          # lines 1..382
tail = "".join(src[SPLIT - 1:])          # lines 383..end
config_and_imports = "".join(src[17:57])  # lines 18..57: config block + imports

# ── Phase A ───────────────────────────────────────────────────────────────────
handoff = '''
# ── Hand-off to stage 08a ─────────────────────────────────────────────────────
# The merge into validation_metrics.csv and the 5-model comparison plots need
# stage 07 to have run, so they live in 08a_dnn_merge_validation.py. Persist what
# that script needs.
import json as _json
os.makedirs(VALD_DIR, exist_ok=True)
with open(os.path.join(VALD_DIR, 'dnn_metrics.json'), 'w') as _fh:
    _json.dump(dnn_metrics, _fh, indent=2)
np.savez(os.path.join(VALD_DIR, 'dnn_test_predictions.npz'),
         y_te=np.asarray(y_te), probs_test=np.asarray(probs_test),
         best_thresh=np.float64(best_thresh))
print("\\nSaved dnn_metrics.json and dnn_test_predictions.npz for stage 08a.")
print("Run 08a_dnn_merge_validation.py after stage 07 to merge the comparison.")
'''

phase_a = head.replace(
    "  ML_Results/validation/  ← updated comparison including DNN",
    "  08_Validation/dnn_metrics.json          metrics handed to stage 08a\n"
    "  08_Validation/dnn_test_predictions.npz  test-set preds handed to stage 08a\n"
    "\n"
    "NOTE: this script stops after computing DNN metrics. The merge into\n"
    "validation_metrics.csv and the 5-model comparison plots require stage 07 and\n"
    "live in 08a_dnn_merge_validation.py.",
) + handoff

# ── Phase B ───────────────────────────────────────────────────────────────────
phase_b_header = '''"""
DNN Validation Merge — Flood Susceptibility Mapping (Slovenia)
=============================================================
Phase B of the former train_dnn.py. Merges the DNN row into
validation_metrics.csv and regenerates the 5-model comparison plots.

MUST run after stage 07 (threshold_classify_validate.py), which creates
validation_metrics.csv. Reads the hand-off files written by 05b_train_dnn.py.

Inputs
  08_Validation/validation_metrics.csv        from stage 07
  08_Validation/dnn_metrics.json              from stage 05b
  08_Validation/dnn_test_predictions.npz      from stage 05b
Outputs
  08_Validation/validation_metrics.csv        updated, now including DNN
  08_Validation/  comparison ROC / PR / bar plots across all 5 models
"""

'''

phase_b_load = '''
# ── Load the hand-off from stage 05b ──────────────────────────────────────────
import json as _json

_metrics_path = os.path.join(VALD_DIR, 'dnn_metrics.json')
_preds_path   = os.path.join(VALD_DIR, 'dnn_test_predictions.npz')
for _p in (_metrics_path, _preds_path):
    if not os.path.exists(_p):
        raise SystemExit(
            f"Missing {_p}\\n"
            "Run 05b_train_dnn.py first (it writes the hand-off files)."
        )
if not os.path.exists(os.path.join(VALD_DIR, 'validation_metrics.csv')):
    raise SystemExit(
        f"Missing {os.path.join(VALD_DIR, 'validation_metrics.csv')}\\n"
        "Run stage 07 (07_threshold_classify_validate.py) first - it creates this file."
    )

with open(_metrics_path) as _fh:
    dnn_metrics = _json.load(_fh)
_z = np.load(_preds_path)
y_te, probs_test, best_thresh = _z['y_te'], _z['probs_test'], float(_z['best_thresh'])
print(f"Loaded DNN hand-off: AUC={dnn_metrics['AUC']}  threshold={best_thresh:.4f}")

'''

phase_b = phase_b_header + config_and_imports + phase_b_load + tail

(DST / "05b_train_dnn.py").write_text(phase_a)
(DST / "08a_dnn_merge_validation.py").write_text(phase_b)

print(f"05b_train_dnn.py          -> {len(phase_a.splitlines())} lines (phase A)")
print(f"08a_dnn_merge_validation.py -> {len(phase_b.splitlines())} lines (phase B)")
