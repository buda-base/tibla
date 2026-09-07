#!/usr/bin/env python3
"""Score the v4 (Hub tag `v4`, 833-page test) prediction dumps under the same
COTe / Hidden-Trespass / LED / COCO protocol as run_literature_eval.py, but over
the v4 system set and with a v4-focused write-up (no DocLayNet-transfer or
curriculum sections — those are v2-paper-specific).

Predictions are the archived YOLO label dumps under --pred-root/<id>/labels
(one dir per system, mirrored from s3://.../tdlav4/eval/... and
off-the-shelf-eval-tdlav4/...). GT is the leak-free v4 833-page test.

Each system's operating point is its own best-mean-F1 confidence (from the
canonical sweep), so Hidden Trespass / COTe / LED are all reported at the same
point as the headline F1 — consistent with the tdla-v4 result tables.

Usage:
  .venv_eval/bin/python evaluation/run_v4_lit_eval.py \\
      --gt-dir  /home/eroux/tmp/dataset_tdlav4_tam2col/labels/test \\
      --img-dir /home/eroux/tmp/dataset_tdlav4_tam2col/images/test \\
      --pred-root /home/eroux/tmp/tdlav4_lit/preds \\
      --out-dir evaluation/eval_results/tdla-v4/literature
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from literature_metrics import (
    best_f1_sweep, coco_map, contamination, cote_dataset, hidden_trespass,
    hidden_trespass_crop, iter_pages, led_errors, load_sizes, operating_points,
)

# id -> (display name, group, role). Order = table order (ours first).
SYSTEMS = [
    ("rtdetr_tdlav4",          "RT-DETR-l tam2col (ours, v4; seed0)",        "ours",          "ours"),
    ("rfdetr_tdlav4",          "RF-DETR-Large tam2col (ours, v4)",           "ours",          "ours"),
    ("docling_heron_tdlav4",   "Docling layout-heron tam2col (ours, v4)",    "ours",          "ours"),
    ("doclayout_tdlav4",       "DocLayout-YOLO tam2col (ours, v4)",          "ours",          "ours"),
    ("pp_doclayout_tdlav4",    "PP-DocLayout-L tam2col (ours, v4)",          "ours",          "ours"),
    ("pp_doclayout_ots",       "PP-DocLayout-L (off-the-shelf)",             "off_the_shelf", "other"),
    ("docling_heron_ots",      "Docling layout-heron (off-the-shelf)",       "off_the_shelf", "doclaynet_detector"),
    ("doclayout_docstruct_ots","DocLayout-YOLO DocStructBench (off-the-shelf)","off_the_shelf","other"),
    ("surya_vlm",              "Surya 2 layout VLM (surya-ocr-2)",           "off_the_shelf", "other"),
    ("chandra",                "Chandra 2 (chandra-ocr-2)",                  "off_the_shelf", "other"),
    ("azure_di",               "Azure DI prebuilt-layout",                   "off_the_shelf", "other"),
    ("aws_textract",           "AWS Textract Layout",                        "off_the_shelf", "other"),
    ("google_docai",           "Google DocAI Layout Parser",                 "off_the_shelf", "other"),
]


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return f"{x:.{nd}f}"


def _pct(x):
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return f"{100 * x:.1f}%"


def score_system(sid, name, group, role, pages, out_dir, skip_cote=False):
    t0 = time.time()
    sweep = best_f1_sweep(pages, "canonical")
    conf = sweep["best_mean_conf"]
    print(f"\n=== {sid}  best-mean conf={conf} ===", flush=True)
    coco = coco_map(pages, "canonical")
    op = operating_points(pages, "canonical", conf)
    result = {
        "id": sid, "name": name, "group": group, "role": role,
        "dataset_tag": "v4", "operating_conf": conf,
        "coco_canonical": coco,
        "sweep_canonical": {
            "best_mean_F1": sweep["best_mean_F1"],
            "best_mean_conf": sweep["best_mean_conf"],
            "best_per_class": sweep["best_per_class"],
            "at_best_mean": sweep["at_best_mean"],
        },
        "op_canonical": op,
        "hidden_trespass": hidden_trespass(pages, conf),
        "hidden_trespass_crop": hidden_trespass_crop(pages, conf),
        "contamination": contamination(pages, conf),
        "led": led_errors(pages, conf),
    }
    if not skip_cote:
        result["cote"] = cote_dataset(pages, conf, max_dim=1024)
    result["wall_s"] = round(time.time() - t0, 2)
    ht = result["hidden_trespass"]
    htc = result["hidden_trespass_crop"]
    print(f"  done {result['wall_s']}s  meanF1={sweep['best_mean_F1']:.3f}  "
          f"HT-hf={ht['header-footer']['HT']:.4f}->crop {htc['header-footer']['HT']:.4f}  "
          f"HT-fn={ht['footnote']['HT']:.4f}->crop {htc['footnote']['HT']:.4f}"
          + ("" if skip_cote else f"  COTe-T={result['cote']['trespass']:.4f}"),
          flush=True)
    (out_dir / f"{sid}.json").write_text(json.dumps(result, indent=2))
    return result


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def write_note(results, meta, path, skip_cote=False):
    L = []
    a = L.append
    a("# tdla-v4 — COTe + Hidden-Trespass evaluation")
    a("")
    a(f"Area-based failure analysis on the leak-free **v4 {meta['n_images']}-page "
      f"test** (`{meta['gt_dir']}`). Same metric code as the v2 literature note "
      "(`literature_metrics.py`); this driver runs it over the v4 system set and "
      "reports each system at its **own best-mean-F1 operating point**. Predictions "
      "are the archived YOLO dumps under `s3://.../tdlav4/eval/...`; nothing was "
      "re-inferred here.")
    a("")
    a("- Canonical 3-class space: header+footer combined (matched individually), "
      "text-area merged to one page/column envelope, footnote as-is; IoU≥0.5.")
    a("- COTe via `cotescore` 0.2.0; the text-area envelope is the body SSU.")
    a(f"- Scoring wall-clock: **{meta['wall_s']:.1f}s** CPU.")
    a("")

    a("## Headline — canonical COCO mAP + F1 @ best-mean point")
    a("")
    a("| system | mean F1 | conf | hf F1 | ta F1 | fn F1 | mean AP50 | mean AP50-95 |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        p = r["op_canonical"]["per_class"]
        a(f"| {r['name']} | {_fmt(r['sweep_canonical']['best_mean_F1'])} | "
          f"{r['operating_conf']:.2f} | {_fmt(p['header-footer']['F1'])} | "
          f"{_fmt(p['text-area']['F1'])} | {_fmt(p['footnote']['F1'])} | "
          f"{_fmt(r['coco_canonical']['mean_ap50'])} | "
          f"{_fmt(r['coco_canonical']['mean_ap5095'])} |")
    a("")

    a("## Hidden Trespass (area-based text-area → clutter bleed)")
    a("")
    a("Primary metric. For each class *c* ∈ {header-footer, footnote}, micro-averaged")
    a("over the test set (Σ area over pages):")
    a("")
    a("- **HT_c** = area(E ∩ U_c) / area(G_c) — *hidden* bleed: class-*c* GT area that")
    a("  is undetected (no same-class pred at IoU≥0.5) yet sits inside the predicted")
    a("  text-area envelope *E* (the OCR crop). Continuous overlap, no 50% cut-off.")
    a("- **R_c** = area(E ∩ D_c) / area(G_c) — *removed-before-OCR*: detected class-*c*")
    a("  GT also inside *E*; a post-processor can punch it back out.")
    a("- **total_c** = HT_c + R_c = area(E ∩ G_c) / area(G_c).")
    a("")
    a("Count-based **contam.** (undetected AND ≥50% absorbed, fraction of *regions*) is")
    a("the secondary intuition column.")
    a("")
    a("| system | hf HT | hf R | hf total | hf contam. (count) | fn HT | fn R | fn total | fn contam. (count) |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        ht = r["hidden_trespass"]
        hf, fn = ht["header-footer"], ht["footnote"]
        a(f"| {r['name']} | {_fmt(hf['HT'])} | {_fmt(hf['R'])} | {_fmt(hf['total_bleed'])} | "
          f"{_pct(hf['count_contamination_rate'])} | {_fmt(fn['HT'])} | {_fmt(fn['R'])} | "
          f"{_fmt(fn['total_bleed'])} | {_pct(fn['count_contamination_rate'])} |")
    a("")

    a("## Post-subtraction crop Hidden Trespass (production body crop)")
    a("")
    a("Exact metric. The OCR body crop is C = E \\ P where E is the predicted")
    a("text-area envelope and P is the geometric union of the retained peripheral")
    a("predictions the production pipeline paints out of the body crop "
      "(**header + footnote + footer**, matching bec-orchestration `paddleocr_v2`")
    a("`crop_body_regions`). For class *c* ∈ {header-footer, footnote},")
    a("micro-averaged over pages:")
    a("")
    a("- **HT_c(crop)** = Σ area(C ∩ g) / Σ area(g) — GT peripheral area that")
    a("  remains in the post-subtraction body crop.")
    a("- decomposition: **matched-residual** (matched but incompletely removed) +")
    a("  **unmatched-residual** (missed GT) = HT_c(crop).")
    a("- **raw** = Σ area(E ∩ g)/Σarea(g) (pre-subtraction); **removed** = area")
    a("  punched out by peripheral preds; raw = HT(crop) + removed.")
    a("- **old HT** = previous match-conditioned metric (undetected area only, no")
    a("  subtraction), shown for comparison at the SAME operating confidence.")
    a("")
    a("| system | conf | old hf HT | new hf HT | Δhf | hf matched | hf unmatched | hf raw | hf removed | old fn HT | new fn HT | Δfn | fn matched | fn unmatched |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        o = r["hidden_trespass"]
        c = r["hidden_trespass_crop"]
        ohf, ofn = o["header-footer"], o["footnote"]
        chf, cfn = c["header-footer"], c["footnote"]
        a(f"| {r['name']} | {r['operating_conf']:.2f} | "
          f"{_fmt(ohf['HT'],4)} | {_fmt(chf['HT'],4)} | {_fmt(chf['HT']-ohf['HT'],4)} | "
          f"{_fmt(chf['matched_residual'],4)} | {_fmt(chf['unmatched_residual'],4)} | "
          f"{_fmt(chf['raw_overlap'],4)} | {_fmt(chf['removed'],4)} | "
          f"{_fmt(ofn['HT'],4)} | {_fmt(cfn['HT'],4)} | {_fmt(cfn['HT']-ofn['HT'],4)} | "
          f"{_fmt(cfn['matched_residual'],4)} | {_fmt(cfn['unmatched_residual'],4)} |")
    a("")

    if not skip_cote:
        a("## COTe decomposition + LED")
        a("")
        a("| system | COTe | Coverage | Overlap | **Trespass** | Excess | ta→peri Trespass | LED-Merge hf | LED-Missing hf | LED-Merge fn | LED-Missing fn |")
        a("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in results:
            c = r["cote"]
            led = r["led"]
            a(f"| {r['name']} | {_fmt(c['cote'])} | {_fmt(c['coverage'])} | "
              f"{_fmt(c['overlap'])} | {_fmt(c['trespass'])} | {_fmt(c['excess'])} | "
              f"{_fmt(c.get('ta_trespass_peripheral'))} | "
              f"{led['header-footer']['merge']} | {led['header-footer']['missing']} | "
              f"{led['footnote']['merge']} | {led['footnote']['missing']} |")
        a("")

    # ---- Spearman cross-check ----
    def ht_combined(r):
        h = r["hidden_trespass"]
        num = h["header-footer"]["area_E_inter_U_px"] + h["footnote"]["area_E_inter_U_px"]
        den = h["header-footer"]["area_G_px"] + h["footnote"]["area_G_px"]
        return num / den if den else 0.0

    def contam_combined(r):
        c = r["contamination"]
        hf, fn = c["header-footer"], c["footnote"]
        n = hf["gt"] + fn["gt"]
        return (hf["absorbed"] + fn["absorbed"]) / n if n else 0.0

    def merge_combined(r):
        return r["led"]["header-footer"]["merge"] + r["led"]["footnote"]["merge"]

    rho = {}
    if not skip_cote:
        hs = [ht_combined(r) for r in results]
        ts = [r["cote"].get("ta_trespass_peripheral", r["cote"]["trespass"]) for r in results]
        ms = [merge_combined(r) for r in results]
        xs = [contam_combined(r) for r in results]
        ys = [r["cote"]["trespass"] for r in results]
        rho = {
            "spearman_ht_trespass": _spearman(hs, ts),
            "spearman_ht_merge": _spearman(hs, ms),
            "spearman_contam_trespass": _spearman(xs, ys),
            "spearman_contam_merge": _spearman(xs, ms),
        }
        a("## Cross-check (Spearman across all systems)")
        a("")
        a(f"Area-based Hidden Trespass vs library metrics, {len(results)} systems: "
          f"ρ(HT, COTe-Trespass text-area→peripheral) = **{_fmt(rho['spearman_ht_trespass'], 3)}**; "
          f"ρ(HT, LED-Merge) = **{_fmt(rho['spearman_ht_merge'], 3)}**.")
        a("")
        a(f"Legacy count-based contamination: ρ(contamination, COTe-Trespass) = "
          f"**{_fmt(rho['spearman_contam_trespass'], 3)}**; ρ(contamination, LED-Merge) = "
          f"**{_fmt(rho['spearman_contam_merge'], 3)}**.")
        a("")

    path.write_text("\n".join(L) + "\n")
    return rho


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--img-dir", type=Path, required=True)
    ap.add_argument("--pred-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--systems", default="")
    ap.add_argument("--skip-cote", action="store_true")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    sizes = load_sizes(args.img_dir, cache=out / "test_image_sizes.json")
    print(f"{len(sizes)} image sizes", flush=True)

    want = {s.strip() for s in args.systems.split(",") if s.strip()}
    t0 = time.time()
    results = []
    for sid, name, group, role in SYSTEMS:
        if want and sid not in want:
            continue
        pred = args.pred_root / sid / "labels"
        if not pred.is_dir():
            print(f"skip {sid}: no pred dir {pred}", file=sys.stderr)
            continue
        pages = list(iter_pages(args.gt_dir, pred, sizes, conf_floor=0.0))
        print(f"{sid}: {len(pages)} pages", flush=True)
        results.append(score_system(sid, name, group, role, pages, out,
                                    skip_cote=args.skip_cote))

    meta = {
        "n_images": len(sizes), "gt_dir": str(args.gt_dir),
        "img_dir": str(args.img_dir), "wall_s": time.time() - t0,
        "dataset_tag": "v4", "cotescore": "0.2.0",
    }
    rho = write_note(results, meta, out / "RESULTS.md", skip_cote=args.skip_cote)
    (out / "metrics.json").write_text(json.dumps(
        {"meta": meta, "rank_agreement": rho, "systems": results}, indent=2))
    print(f"\nwrote {out/'metrics.json'} and {out/'RESULTS.md'} in "
          f"{meta['wall_s']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
