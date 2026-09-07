#!/usr/bin/env python3
"""Unit tests for the exact post-subtraction crop Hidden Trespass metric.

Run directly (no pytest needed):
    .venv_eval/bin/python evaluation/test_hidden_trespass.py
or with pytest if available:
    pytest evaluation/test_hidden_trespass.py

Covers the six required cases:
  1. 100-px header inside E, matched pred covers 60  -> residual 40.
  2. A completely missed region.
  3. Overlapping peripheral predictions (overlap subtracted only once).
  4. No predicted text-area.
  5. A false-positive peripheral prediction overlapping a GT peripheral region.
  6. A matched prediction that completely removes the region.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from literature_metrics import (  # noqa: E402
    crop_residual, rect_union_area, hidden_trespass_crop,
)

W = H = 1000  # test canvas


def _box(cls, x1, y1, x2, y2, conf=1.0):
    """Build a YOLO-dict box (as read_yolo would) from pixel corners."""
    return {"cls": cls, "conf": conf,
            "xyxyn": [x1 / W, y1 / H, x2 / W, y2 / H],
            "xywhn": [(x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
                      (x2 - x1) / W, (y2 - y1) / H]}


def _page(gt, pred, stem="p"):
    return (stem, W, H, gt, pred)


ASSERTS = 0


def _eq(a, b, msg, tol=1e-6):
    global ASSERTS
    ASSERTS += 1
    assert abs(a - b) <= tol, f"{msg}: expected {b}, got {a}"


# --- geometry-level tests (crop_residual / rect_union_area) -----------------

def test_case1_matched_covers_60_residual_40():
    """100-px header inside E, matched pred covers 60 px -> residual 40."""
    E = [0, 0, 100, 100]
    g = [0, 0, 10, 10]                 # area 100
    pred = [0, 0, 6, 10]               # covers 60 of the 100
    e_inter, removed, residual = crop_residual(E, [pred], g)
    _eq(e_inter, 100.0, "case1 E∩g area")
    _eq(removed, 60.0, "case1 removed")
    _eq(residual, 40.0, "case1 residual")


def test_case2_completely_missed():
    """A completely missed region: no peripheral pred -> full area residual."""
    E = [0, 0, 100, 100]
    g = [0, 0, 10, 10]                 # area 100
    e_inter, removed, residual = crop_residual(E, [], g)
    _eq(e_inter, 100.0, "case2 E∩g area")
    _eq(removed, 0.0, "case2 removed")
    _eq(residual, 100.0, "case2 residual (whole region remains)")


def test_case3_overlapping_peripheral_subtracted_once():
    """Overlapping peripheral predictions: overlap counted only once."""
    E = [0, 0, 100, 100]
    g = [0, 0, 10, 10]                 # area 100
    p1 = [0, 0, 6, 10]                 # 60
    p2 = [4, 0, 8, 10]                 # 40, overlaps p1 on x∈[4,6] (20)
    # naive sum would be 100; true union over g is x∈[0,8] -> 80.
    _eq(rect_union_area([[0, 0, 6, 10], [4, 0, 8, 10]]), 80.0,
        "case3 raw union area")
    e_inter, removed, residual = crop_residual(E, [p1, p2], g)
    _eq(removed, 80.0, "case3 removed (union once, not 100)")
    _eq(residual, 20.0, "case3 residual")


def test_case4_no_text_area():
    """No predicted text-area (E is None) -> zero residual everywhere."""
    g = [0, 0, 10, 10]
    pred = [0, 0, 6, 10]
    e_inter, removed, residual = crop_residual(None, [pred], g)
    _eq(e_inter, 0.0, "case4 E∩g")
    _eq(removed, 0.0, "case4 removed")
    _eq(residual, 0.0, "case4 residual")


# --- end-to-end tests (hidden_trespass_crop, matched/unmatched split) -------

def test_case1_endToEnd_matched_split():
    """Same as case 1 but through hidden_trespass_crop: matched, 40 residual."""
    gt = [_box(0, 0, 0, 10, 10)]                          # header GT area 100
    pred = [_box(1, 0, 0, 100, 100, 0.9),                 # text-area envelope
            _box(0, 0, 0, 6, 10, 0.9)]                    # header pred (IoU .6)
    out = hidden_trespass_crop([_page(gt, pred)], conf=0.0)
    hf = out["header-footer"]
    _eq(hf["num_residual_px"], 40.0, "e2e case1 residual px")
    _eq(hf["num_raw_px"], 100.0, "e2e case1 raw overlap px")
    _eq(hf["num_removed_px"], 60.0, "e2e case1 removed px")
    _eq(hf["n_matched"], 1, "e2e case1 matched count")
    _eq(hf["n_unmatched"], 0, "e2e case1 unmatched count")
    # HT = matched_residual + unmatched_residual (identity)
    _eq(hf["HT"], hf["matched_residual"] + hf["unmatched_residual"],
        "e2e case1 decomposition identity")
    _eq(hf["matched_residual"], hf["HT"], "e2e case1 all residual is matched")


def test_case4_endToEnd_no_text_area():
    """No text-area predicted -> HT 0, page flagged, GT still counted."""
    gt = [_box(0, 0, 0, 10, 10)]
    pred = [_box(0, 0, 0, 6, 10, 0.9)]                    # header pred, NO TA
    out = hidden_trespass_crop([_page(gt, pred)], conf=0.0)
    hf = out["header-footer"]
    _eq(hf["HT"], 0.0, "e2e case4 HT is 0 with no TA")
    _eq(hf["num_residual_px"], 0.0, "e2e case4 residual px 0")
    _eq(hf["n_pages_no_ta"], 1, "e2e case4 no-TA page flagged")
    _eq(hf["n_gt"], 1, "e2e case4 GT still counted in denominator")


def test_case5_false_positive_peripheral_unmatched_still_removes():
    """FP peripheral (IoU<0.5, so unmatched) still punches its area out."""
    gt = [_box(0, 0, 0, 20, 20)]                          # header GT area 400
    pred = [_box(1, 0, 0, 100, 100, 0.9),                 # text-area envelope
            _box(0, 0, 0, 4, 20, 0.9)]                    # header pred area 80
    #                                                       IoU = 80/400 = 0.2 -> unmatched
    out = hidden_trespass_crop([_page(gt, pred)], conf=0.0)
    hf = out["header-footer"]
    _eq(hf["n_matched"], 0, "e2e case5 no match (IoU<0.5)")
    _eq(hf["n_unmatched"], 1, "e2e case5 unmatched")
    _eq(hf["num_removed_px"], 80.0, "e2e case5 FP still removed 80")
    _eq(hf["num_residual_px"], 320.0, "e2e case5 residual 320")
    _eq(hf["unmatched_residual"] * hf["area_G_px"], 320.0,
        "e2e case5 residual attributed to unmatched")


def test_case6_matched_complete_removal():
    """Matched pred fully covers the region -> residual 0, matched."""
    gt = [_box(3, 0, 0, 10, 10)]                          # footer GT area 100
    pred = [_box(1, 0, 0, 100, 100, 0.9),                 # text-area envelope
            _box(3, 0, 0, 10, 10, 0.9)]                   # footer pred IoU 1.0
    out = hidden_trespass_crop([_page(gt, pred)], conf=0.0)
    hf = out["header-footer"]
    _eq(hf["n_matched"], 1, "e2e case6 matched")
    _eq(hf["num_removed_px"], 100.0, "e2e case6 fully removed")
    _eq(hf["num_residual_px"], 0.0, "e2e case6 residual 0")
    _eq(hf["HT"], 0.0, "e2e case6 HT 0")


def test_decomposition_identity_mixed_page():
    """Mixed page: raw_overlap = HT + removed, and HT = matched+unmatched."""
    gt = [_box(0, 0, 0, 10, 10),          # header, will be matched (removed 60)
          _box(3, 0, 20, 10, 30),         # footer, missed (no pred) -> unmatched
          _box(2, 0, 40, 10, 50)]         # footnote, matched fully
    pred = [_box(1, 0, 0, 100, 100, 0.9),                 # TA envelope
            _box(0, 0, 0, 6, 10, 0.9),                    # header pred IoU .6
            _box(2, 0, 40, 10, 50, 0.9)]                  # footnote pred IoU 1
    out = hidden_trespass_crop([_page(gt, pred)], conf=0.0)
    hf, fn = out["header-footer"], out["footnote"]
    # header-footer: header residual 40 (matched), footer residual 100 (unmatched)
    _eq(hf["matched_residual"] * hf["area_G_px"], 40.0, "mixed hf matched res px")
    _eq(hf["unmatched_residual"] * hf["area_G_px"], 100.0, "mixed hf unmatched res px")
    _eq(hf["HT"] * hf["area_G_px"], 140.0, "mixed hf HT px")
    _eq(hf["raw_overlap"], hf["HT"] + hf["removed"], "mixed hf raw=HT+removed")
    # footnote: fully removed -> HT 0
    _eq(fn["num_residual_px"], 0.0, "mixed fn residual 0")
    _eq(fn["n_matched"], 1, "mixed fn matched")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed, "
          f"{ASSERTS} assertions.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
