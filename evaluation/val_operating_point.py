#!/usr/bin/env python3
"""Validation-based operating-point selection + leak-free Hidden-Trespass rows.

For each thresholdable system:
  1. Sweep the global confidence on the VALIDATION split (canonical 3-class
     macro mean F1, 0.01 grid — identical rule to the test sweep) and pick the
     argmax as `val_conf`. This never touches the test set.
  2. Freeze `val_conf`; on the TEST split report canonical mean F1, the previous
     (match-conditioned) Hidden Trespass, and the new post-subtraction crop
     Hidden Trespass.

Decomposition reported per system (separates the metric change from the
threshold change):
  row1 = existing HT   @ old test-selected conf   (from literature/metrics.json)
  row2 = post-crop HT  @ old test-selected conf   (from literature/metrics.json)
  row3 = existing HT   @ val-selected conf
  row4 = post-crop HT  @ val-selected conf         <- final paper value

Selection rule (exact): val_conf = argmax over c in {0.00,0.01,...,0.99} of the
canonical macro mean F1 = mean(F1_header-footer, F1_text-area, F1_footnote) on
the validation split, greedy IoU>=0.5 matching, text-area merged to one
envelope built from predictions kept at >= c. Ties broken by the lowest conf.

Systems without validation predictions (commercial APIs, VLMs — no usable
per-box confidence to sweep) retain their fixed operating point: val_conf = the
existing operating conf, and rows 3/4 equal rows 1/2.

Usage:
  .venv_eval/bin/python evaluation/val_operating_point.py \
      --val-gt   /home/eroux/tmp/dataset_tdlav4_tam2col/labels/val \
      --val-img  /home/eroux/tmp/dataset_tdlav4_tam2col/images/val \
      --test-gt  /home/eroux/tmp/dataset_tdlav4_tam2col/labels/test \
      --test-img /home/eroux/tmp/dataset_tdlav4_tam2col/images/test \
      --val-pred-root  /home/eroux/tmp/tdlav4_val/preds \
      --test-metrics   evaluation/eval_results/tdla-v4/literature/metrics.json \
      --test-pred-root /home/eroux/tmp/tdlav4_lit/preds \
      --out-dir  evaluation/eval_results/tdla-v4/literature
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_metrics import (
    best_f1_sweep, hidden_trespass, hidden_trespass_crop, iter_pages,
    load_sizes, operating_points,
)

# The 8 thresholdable systems for which we generate validation dumps.
THRESHOLDABLE = [
    "rtdetr_tdlav4", "rfdetr_tdlav4", "docling_heron_tdlav4",
    "doclayout_tdlav4", "pp_doclayout_tdlav4",
    "pp_doclayout_ots", "docling_heron_ots", "doclayout_docstruct_ots",
]


def _ht_pair(pages, conf):
    o = hidden_trespass(pages, conf)
    c = hidden_trespass_crop(pages, conf)
    return {
        "hf_old": o["header-footer"]["HT"], "hf_new": c["header-footer"]["HT"],
        "fn_old": o["footnote"]["HT"],       "fn_new": c["footnote"]["HT"],
        "hf_matched_residual": c["header-footer"]["matched_residual"],
        "hf_unmatched_residual": c["header-footer"]["unmatched_residual"],
        "fn_matched_residual": c["footnote"]["matched_residual"],
        "fn_unmatched_residual": c["footnote"]["unmatched_residual"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-gt", type=Path, required=True)
    ap.add_argument("--val-img", type=Path, required=True)
    ap.add_argument("--test-gt", type=Path, required=True)
    ap.add_argument("--test-img", type=Path, required=True)
    ap.add_argument("--val-pred-root", type=Path, required=True)
    ap.add_argument("--test-pred-root", type=Path, required=True)
    ap.add_argument("--test-metrics", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    val_sizes = load_sizes(args.val_img, cache=out / "val_image_sizes.json")
    test_sizes = load_sizes(args.test_img, cache=out / "test_image_sizes.json")
    print(f"{len(val_sizes)} val / {len(test_sizes)} test image sizes", flush=True)

    tm = {s["id"]: s for s in json.loads(args.test_metrics.read_text())["systems"]}

    results = []
    for sid, s in tm.items():
        old_conf = s["operating_conf"]
        row12 = {
            "hf_old": s["hidden_trespass"]["header-footer"]["HT"],
            "hf_new": s["hidden_trespass_crop"]["header-footer"]["HT"],
            "fn_old": s["hidden_trespass"]["footnote"]["HT"],
            "fn_new": s["hidden_trespass_crop"]["footnote"]["HT"],
            "hf_matched_residual": s["hidden_trespass_crop"]["header-footer"]["matched_residual"],
            "hf_unmatched_residual": s["hidden_trespass_crop"]["header-footer"]["unmatched_residual"],
            "fn_matched_residual": s["hidden_trespass_crop"]["footnote"]["matched_residual"],
            "fn_unmatched_residual": s["hidden_trespass_crop"]["footnote"]["unmatched_residual"],
        }
        rec = {"id": sid, "name": s["name"], "group": s["group"],
               "old_test_conf": old_conf,
               "test_F1_old_conf": s["sweep_canonical"]["best_mean_F1"],
               "row1_2_at_old_conf": row12}

        val_pred = args.val_pred_root / sid / "labels"
        if sid not in THRESHOLDABLE or not val_pred.is_dir() or not any(val_pred.glob("*.txt")):
            rec["val_available"] = False
            rec["val_conf"] = old_conf
            rec["note"] = "fixed operating point (no usable confidence / no val dump)"
            rec["row3_4_at_val_conf"] = row12  # identical: same conf
            results.append(rec)
            print(f"{sid:26} FIXED-OP (val n/a)  conf={old_conf:.2f}", flush=True)
            continue

        # --- validation sweep (selection) ---
        val_pages = list(iter_pages(args.val_gt, val_pred, val_sizes, conf_floor=0.0))
        sweep = best_f1_sweep(val_pages, "canonical")
        val_conf = sweep["best_mean_conf"]
        val_op = sweep["at_best_mean"]
        # per-class validation curve (for deployment thresholds; no test tuning)
        curve = []
        # rebuild curve at a coarse grid for the note
        for c in [round(0.05 * i, 2) for i in range(1, 20)]:
            op = operating_points(val_pages, "canonical", c)
            pc = op["per_class"]
            curve.append({"conf": c, "mean_F1": op["mean_F1"],
                          "hf_F1": pc["header-footer"]["F1"],
                          "hf_P": pc["header-footer"]["P"], "hf_R": pc["header-footer"]["R"],
                          "ta_F1": pc["text-area"]["F1"],
                          "ta_P": pc["text-area"]["P"], "ta_R": pc["text-area"]["R"],
                          "fn_F1": pc["footnote"]["F1"],
                          "fn_P": pc["footnote"]["P"], "fn_R": pc["footnote"]["R"]})

        # --- frozen test evaluation at val_conf ---
        test_pred = args.test_pred_root / sid / "labels"
        test_pages = list(iter_pages(args.test_gt, test_pred, test_sizes, conf_floor=0.0))
        test_op = operating_points(test_pages, "canonical", val_conf)
        row34 = _ht_pair(test_pages, val_conf)

        rec.update({
            "val_available": True,
            "val_conf": val_conf,
            "val_meanF1": sweep["best_mean_F1"],
            "val_per_class_F1": {k: v["F1"] for k, v in val_op["per_class"].items()},
            "test_F1_at_val_conf": test_op["mean_F1"],
            "test_per_class_F1_at_val_conf": {k: v["F1"] for k, v in test_op["per_class"].items()},
            "row3_4_at_val_conf": row34,
            "val_curve": curve,
        })
        results.append(rec)
        print(f"{sid:26} val_conf={val_conf:.2f} valF1={sweep['best_mean_F1']:.3f} "
              f"testF1@val={test_op['mean_F1']:.3f} (old test conf {old_conf:.2f})  "
              f"HTfn {row12['fn_new']:.4f}->{row34['fn_new']:.4f}", flush=True)

    (out / "val_operating_point.json").write_text(json.dumps(
        {"selection_rule": "argmax canonical macro mean F1 on val, 0.01 grid, "
                           "ties->lowest conf; text-area envelope from kept preds",
         "thresholdable": THRESHOLDABLE, "systems": results}, indent=2))
    print(f"\nwrote {out/'val_operating_point.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
