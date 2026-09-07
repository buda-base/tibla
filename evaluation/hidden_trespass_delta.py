#!/usr/bin/env python3
"""Emit the old-vs-new (post-subtraction crop) Hidden Trespass comparison.

Reads the literature metrics.json produced by run_v4_lit_eval.py (which now
carries both `hidden_trespass` — the previous match-conditioned metric — and
`hidden_trespass_crop` — the exact post-subtraction body-crop metric) and
writes a focused machine-readable JSON plus a markdown delta table.

Both metrics are reported at the SAME (paper) operating confidence per system,
so the delta isolates the metric change (no threshold retuning).

Usage:
  .venv_eval/bin/python evaluation/hidden_trespass_delta.py \\
      --metrics evaluation/eval_results/tdla-v4/literature/metrics.json \\
      --out     evaluation/eval_results/tdla-v4/literature/hidden_trespass_postcrop.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _row(o, c):
    return {
        "old_HT": o["HT"],
        "new_HT_crop": c["HT"],
        "delta": c["HT"] - o["HT"],
        "matched_residual": c["matched_residual"],
        "unmatched_residual": c["unmatched_residual"],
        "raw_overlap": c["raw_overlap"],
        "removed": c["removed"],
        "n_gt": c["n_gt"],
        "n_matched": c["n_matched"],
        "n_unmatched": c["n_unmatched"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    m = json.loads(args.metrics.read_text())
    out = {
        "note": ("Post-subtraction crop Hidden Trespass vs previous "
                 "match-conditioned HT, at each system's paper operating "
                 "confidence (metric change only, no threshold retuning). "
                 "Body crop C = E \\ P subtracts header+footnote+footer "
                 "predictions (bec-orchestration paddleocr_v2 crop_body_regions)."),
        "subtract_classes": ["header", "footnote", "footer"],
        "systems": [],
    }
    print(f"{'system':38}{'conf':>5} | {'hf old':>7}{'hf new':>7}{'hf Δ':>8} | "
          f"{'fn old':>7}{'fn new':>7}{'fn Δ':>8}")
    print("-" * 96)
    for s in m["systems"]:
        o, c = s["hidden_trespass"], s["hidden_trespass_crop"]
        hf = _row(o["header-footer"], c["header-footer"])
        fn = _row(o["footnote"], c["footnote"])
        out["systems"].append({
            "id": s["id"], "name": s["name"], "group": s["group"],
            "operating_conf": s["operating_conf"],
            "header_footer": hf, "footnote": fn,
        })
        print(f"{s['name'][:37]:38}{s['operating_conf']:5.2f} | "
              f"{hf['old_HT']:7.4f}{hf['new_HT_crop']:7.4f}{hf['delta']:+8.4f} | "
              f"{fn['old_HT']:7.4f}{fn['new_HT_crop']:7.4f}{fn['delta']:+8.4f}")
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
