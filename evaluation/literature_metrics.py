#!/usr/bin/env python3
"""COCO / DocLayNet-protocol metrics, contamination, COTe, and LED error types.

All functions consume YOLO-format boxes (native 4-class ids 0 header, 1
text-area, 2 footnote, 3 footer) plus per-image pixel sizes. Two class schemas:

  (a) canonical  — header+footer relabelled as one class (boxes kept separate),
                   text-area merged to one envelope per page, footnote as-is.
  (b) doclaynet  — page-header / page-footer kept separate, text-area left as
                   native boxes (no envelope), footnote as-is. COCO matching,
                   no post-processing merge.

COCO mAP is pycocotools COCOeval (101-point interpolation, IoU .50:.05:.95,
area=all, maxDets=100), i.e. the same protocol DocLayNet Table 2 reports.
"""
from __future__ import annotations

import io
import json
import math
from contextlib import redirect_stdout
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

NATIVE = {0: "header", 1: "text-area", 2: "footnote", 3: "footer"}
CANON = {0: "header-footer", 1: "text-area", 2: "footnote"}
DOCLAYNET = {0: "page-header", 1: "text-area", 2: "footnote", 3: "page-footer"}
SHARED = ("page-header", "page-footer", "footnote")  # comparable across corpora
MERGE_CANON = {1}  # text-area envelope in schema (a)
IOUS_OP = 0.5
COCO_IOUS = [0.5 + 0.05 * i for i in range(10)]
SWEEP = [round(0.01 * i, 2) for i in range(0, 100)]  # 0.00 .. 0.99


# ---------------------------------------------------------------------------
# geometry / I/O
# ---------------------------------------------------------------------------

def iou_xyxy(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def covered_frac(box, env) -> float:
    if env is None:
        return 0.0
    ix1, iy1 = max(box[0], env[0]), max(box[1], env[1])
    ix2, iy2 = min(box[2], env[2]), min(box[3], env[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area = (box[2] - box[0]) * (box[3] - box[1])
    return inter / area if area > 0 else 0.0


def inter_area(box, env) -> float:
    """Absolute intersection area of `box` and `env` (0 if env is None)."""
    if env is None:
        return 0.0
    ix1, iy1 = max(box[0], env[0]), max(box[1], env[1])
    ix2, iy2 = min(box[2], env[2]), min(box[3], env[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    return iw * ih


def box_area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def rect_inter(a, b):
    """Intersection rectangle of two xyxy boxes, or None if empty/degenerate."""
    if a is None or b is None:
        return None
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def rect_union_area(rects) -> float:
    """Exact area of the union of axis-aligned xyxy rectangles.

    Coordinate-compression sweep: for every x-slab between consecutive unique
    x-edges, sum the length of the y-intervals covered by any rectangle that
    spans the slab. Overlaps are counted once, so this is the geometrically
    exact union area (no double counting).
    """
    rs = [r for r in rects if r is not None and r[2] > r[0] and r[3] > r[1]]
    if not rs:
        return 0.0
    xs = sorted({r[0] for r in rs} | {r[2] for r in rs})
    area = 0.0
    for i in range(len(xs) - 1):
        x1, x2 = xs[i], xs[i + 1]
        dx = x2 - x1
        if dx <= 0:
            continue
        spans = sorted((r[1], r[3]) for r in rs if r[0] <= x1 and r[2] >= x2)
        if not spans:
            continue
        cy = -math.inf
        cov = 0.0
        for a, b in spans:
            if b <= cy:
                continue
            a = max(a, cy)
            cov += b - a
            cy = b
        area += dx * cov
    return area


def crop_residual(E, peripheral_rects, gt_rect):
    """Exact post-subtraction crop residual for one GT rectangle.

    Given the predicted text-area envelope ``E`` (xyxy or None), the list of
    retained peripheral prediction rectangles ``peripheral_rects`` whose union
    is subtracted from the crop, and one ground-truth peripheral box
    ``gt_rect``, return ``(e_inter_g, removed, residual)`` where:

      e_inter_g = area(E ∩ g)                       raw pre-subtraction overlap
      removed   = area(E ∩ g ∩ union(peripheral))    area punched out (once)
      residual  = e_inter_g - removed = area((E \\ P) ∩ g)

    ``E \\ P`` is the actual OCR body crop C. Subtraction of the peripheral
    union is exact (overlapping peripheral predictions are subtracted only
    once). Pages with no predicted text-area pass ``E=None`` -> all zeros.
    """
    R = rect_inter(E, gt_rect)
    if R is None:
        return 0.0, 0.0, 0.0
    e_inter_g = box_area(R)
    clipped = [rect_inter(R, p) for p in peripheral_rects]
    removed = rect_union_area(clipped)
    # numerical guard: removed can never exceed the enclosing rectangle area
    removed = min(removed, e_inter_g)
    return e_inter_g, removed, e_inter_g - removed


def envelope_xyxy(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def yolo_to_xyxy(cx, cy, w, h, W, H):
    return [(cx - w / 2) * W, (cy - h / 2) * H,
            (cx + w / 2) * W, (cy + h / 2) * H]


def read_yolo(path: Path, conf_floor: float = 0.0):
    """List of dicts: cls, conf, xyxy-normalized (x1,y1,x2,y2 in [0,1])."""
    out = []
    if not path.exists():
        return out
    for ln in path.read_text().splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        cls = int(p[0])
        cx, cy, w, h = (float(x) for x in p[1:5])
        conf = float(p[5]) if len(p) >= 6 else 1.0
        if conf < conf_floor:
            continue
        out.append({
            "cls": cls, "conf": conf,
            "xywhn": [cx, cy, w, h],
            "xyxyn": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
        })
    return out


def to_px(box, W, H):
    x1, y1, x2, y2 = box["xyxyn"]
    return [x1 * W, y1 * H, x2 * W, y2 * H]


def load_sizes(img_dir: Path, cache: Path | None = None) -> dict:
    if cache and cache.exists():
        return {k: tuple(v) for k, v in json.loads(cache.read_text()).items()}
    sizes = {}
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}:
            continue
        with Image.open(p) as im:
            sizes[p.stem] = (im.size[0], im.size[1])
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(sizes))
    return sizes


# ---------------------------------------------------------------------------
# schema transforms
# ---------------------------------------------------------------------------

def apply_schema(boxes, schema: str, W: int, H: int, conf_floor: float = 0.0):
    """Return {class_name: [(xyxy_px, conf), ...]}.

    schema 'canonical': header+footer -> header-footer (no envelope);
                        text-area -> one envelope (max conf);
                        footnote unchanged.
    schema 'doclaynet': 4 native classes, no merge.

    `conf_floor` drops boxes below the threshold BEFORE grouping/enveloping.
    This matters for the canonical text-area envelope: at a given operating
    point the OCR crop must be built only from predictions the model would
    actually keep, so sub-threshold body boxes do not inflate the envelope
    (matches eval_pred_files.py / the headline sweep, and is consistent with
    the confidence gating already used by hidden_trespass / contamination /
    cote / led). GT boxes carry conf 1.0 and are unaffected.
    """
    px = []
    for b in boxes:
        if b.get("conf", 1.0) < conf_floor:
            continue
        px.append({**b, "xyxy": to_px(b, W, H)})
    if schema == "canonical":
        grouped = {n: [] for n in CANON.values()}
        ta = [b for b in px if b["cls"] == 1]
        for b in px:
            if b["cls"] in (0, 3):
                grouped["header-footer"].append((b["xyxy"], b["conf"]))
            elif b["cls"] == 2:
                grouped["footnote"].append((b["xyxy"], b["conf"]))
        if ta:
            grouped["text-area"].append(
                (envelope_xyxy([b["xyxy"] for b in ta]),
                 max(b["conf"] for b in ta)))
        return grouped
    if schema == "doclaynet":
        grouped = {n: [] for n in DOCLAYNET.values()}
        for b in px:
            name = DOCLAYNET.get(b["cls"])
            if name is not None:
                grouped[name].append((b["xyxy"], b["conf"]))
        return grouped
    raise ValueError(schema)


def class_names(schema: str):
    return list(CANON.values()) if schema == "canonical" else list(DOCLAYNET.values())


# ---------------------------------------------------------------------------
# COCO mAP (pycocotools)
# ---------------------------------------------------------------------------

def _coco_eval(images, categories, gt_anns, dt_anns):
    if not gt_anns:
        return {c["name"]: {"ap50": float("nan"), "ap5095": float("nan")}
                for c in categories}, float("nan"), float("nan")
    with redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = {
            "info": {}, "licenses": [],
            "images": images, "annotations": gt_anns, "categories": categories,
        }
        coco_gt.createIndex()
        if not dt_anns:
            empty = {c["name"]: {"ap50": 0.0, "ap5095": 0.0} for c in categories}
            return empty, 0.0, 0.0
        coco_dt = coco_gt.loadRes(dt_anns)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.useCats = 1
        ev.evaluate()
        ev.accumulate()
    # suppress summarize() stdout by not calling it; read eval tensor
    prec = ev.eval["precision"]  # T, R, K, A, M
    per = {}
    ap50s, aps = [], []
    for k, cat in enumerate(categories):
        s = prec[:, :, k, 0, 2]
        s = s[s > -1]
        ap = float(s.mean()) if s.size else float("nan")
        s50 = prec[0, :, k, 0, 2]
        s50 = s50[s50 > -1]
        ap50 = float(s50.mean()) if s50.size else float("nan")
        per[cat["name"]] = {"ap50": ap50, "ap5095": ap}
        if not math.isnan(ap50):
            ap50s.append(ap50)
        if not math.isnan(ap):
            aps.append(ap)
    mean50 = float(np.mean(ap50s)) if ap50s else float("nan")
    mean = float(np.mean(aps)) if aps else float("nan")
    return per, mean50, mean


def coco_map(pages, schema: str, mean_over: Iterable[str] | None = None):
    """pages: iterable of (stem, W, H, gt_boxes, pred_boxes) native YOLO dicts.

    mean_over: if set, mean AP is averaged only over those class names
    (used for shared-class domain transfer, excluding text-area).
    """
    names = class_names(schema)
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(names)]
    name_to_id = {n: i + 1 for i, n in enumerate(names)}
    images, gt_anns, dt_anns = [], [], []
    ann_id = 1
    for img_id, (stem, W, H, gt, pred) in enumerate(pages, 1):
        images.append({"id": img_id, "file_name": f"{stem}.jpg",
                       "width": W, "height": H})
        g = apply_schema(gt, schema, W, H)
        p = apply_schema(pred, schema, W, H)
        for name in names:
            cid = name_to_id[name]
            for xyxy, _ in g[name]:
                x1, y1, x2, y2 = xyxy
                bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
                gt_anns.append({
                    "id": ann_id, "image_id": img_id, "category_id": cid,
                    "bbox": [x1, y1, bw, bh], "area": bw * bh, "iscrowd": 0,
                    "ignore": 0, "segmentation": [],
                })
                ann_id += 1
            for xyxy, conf in p[name]:
                x1, y1, x2, y2 = xyxy
                bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
                dt_anns.append({
                    "image_id": img_id, "category_id": cid,
                    "bbox": [x1, y1, bw, bh], "score": float(conf),
                })
    per, mean50, mean = _coco_eval(images, cats, gt_anns, dt_anns)
    if mean_over is not None:
        vals50, vals = [], []
        for n in mean_over:
            if n not in per:
                continue
            if not math.isnan(per[n]["ap50"]):
                vals50.append(per[n]["ap50"])
            if not math.isnan(per[n]["ap5095"]):
                vals.append(per[n]["ap5095"])
        mean50 = float(np.mean(vals50)) if vals50 else float("nan")
        mean = float(np.mean(vals)) if vals else float("nan")
    return {
        "per_class": per,
        "mean_ap50": mean50,
        "mean_ap5095": mean,
        "n_gt": len(gt_anns),
        "n_dt": len(dt_anns),
        "n_images": len(images),
    }


# ---------------------------------------------------------------------------
# operating-point P/R/F1 (greedy IoU>=0.5, same matching as canon_eval.py)
# ---------------------------------------------------------------------------

def _pr_class(gt_list, pred_list, conf, thr=IOUS_OP):
    npos = sum(len(v) for v in gt_list.values())
    preds = []
    for img, items in pred_list.items():
        for xyxy, c in items:
            if c >= conf:
                preds.append((img, xyxy, c))
    preds.sort(key=lambda x: -x[2])
    matched = {img: [False] * len(bs) for img, bs in gt_list.items()}
    tp = fp = 0
    ious = []
    for img, box, _ in preds:
        best, bj = 0.0, -1
        for j, g in enumerate(gt_list.get(img, [])):
            if matched[img][j]:
                continue
            v = iou_xyxy(box, g)
            if v > best:
                best, bj = v, j
        if best >= thr and bj >= 0:
            tp += 1
            matched[img][bj] = True
            ious.append(best)
        else:
            fp += 1
    fn = npos - tp
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    miou = sum(ious) / len(ious) if ious else 0.0
    return {"P": P, "R": R, "F1": F1, "meanIoU": miou,
            "tp": tp, "fp": fp, "fn": fn, "npos": npos}


def operating_points(pages, schema: str, conf: float):
    names = class_names(schema)
    gt = {n: {} for n in names}
    pred = {n: {} for n in names}
    for stem, W, H, gboxes, pboxes in pages:
        g = apply_schema(gboxes, schema, W, H)
        # gate predictions at the operating point BEFORE enveloping, so the
        # text-area envelope reflects only kept boxes.
        p = apply_schema(pboxes, schema, W, H, conf_floor=conf)
        for n in names:
            if g[n]:
                gt[n][stem] = [xy for xy, _ in g[n]]
            if p[n]:
                pred[n][stem] = p[n]
    per = {n: _pr_class(gt[n], pred[n], conf=0.0) for n in names}
    f1s = [per[n]["F1"] for n in names]
    return {"conf": conf, "per_class": per,
            "mean_F1": float(np.mean(f1s)) if f1s else 0.0}


def iou_recall_diagnostic(pages, schema: str, classes, ious=(0.1, 0.5),
                          conf: float = 0.0):
    """Box-convention diagnostic for domain transfer.

    For each class name in `classes` (must exist in `schema`), report at each
    IoU threshold the recall / coverage (fraction of GT with a same-class pred
    matched greedily at IoU>=thr) and the mean IoU over matched pairs. Comparing
    IoU>=0.1 vs 0.5 separates loose localisation (found but boxes off) from
    genuine misses; comparing in-domain vs off-domain mean IoU quantifies the
    box-convention gap. Predictions below `conf` are dropped (pages are usually
    already conf-floored upstream, so the default keeps everything).
    """
    gt = {n: {} for n in classes}
    pred = {n: {} for n in classes}
    for stem, W, H, gboxes, pboxes in pages:
        g = apply_schema(gboxes, schema, W, H)
        p = apply_schema(pboxes, schema, W, H)
        for n in classes:
            if g.get(n):
                gt[n][stem] = [xy for xy, _ in g[n]]
            if p.get(n):
                pred[n][stem] = p[n]
    out = {}
    for n in classes:
        out[n] = {}
        for thr in ious:
            r = _pr_class(gt[n], pred[n], conf=conf, thr=thr)
            out[n][f"iou{thr}"] = {
                "recall": r["R"], "mean_matched_iou": r["meanIoU"],
                "tp": r["tp"], "npos": r["npos"],
            }
    return out


def best_f1_sweep(pages, schema: str, grid=None):
    grid = grid if grid is not None else SWEEP
    names = class_names(schema)
    # GT is confidence-independent -> materialise once. Predictions must be
    # re-gated + re-enveloped per confidence (the text-area envelope shrinks as
    # sub-threshold body boxes drop out), so cache raw pred boxes and rebuild.
    gt = {n: {} for n in names}
    page_preds = []
    for stem, W, H, gboxes, pboxes in pages:
        g = apply_schema(gboxes, schema, W, H)
        for n in names:
            if g[n]:
                gt[n][stem] = [xy for xy, _ in g[n]]
        page_preds.append((stem, W, H, pboxes))
    best_mean = (-1.0, None)
    per_class_best = {n: (-1.0, None) for n in names}
    at = {}
    for conf in grid:
        pred = {n: {} for n in names}
        for stem, W, H, pboxes in page_preds:
            p = apply_schema(pboxes, schema, W, H, conf_floor=conf)
            for n in names:
                if p[n]:
                    pred[n][stem] = p[n]
        per = {n: _pr_class(gt[n], pred[n], conf=0.0) for n in names}
        mean = float(np.mean([per[n]["F1"] for n in names]))
        at[conf] = {"per_class": per, "mean_F1": mean}
        if mean > best_mean[0]:
            best_mean = (mean, conf)
        for n in names:
            if per[n]["F1"] > per_class_best[n][0]:
                per_class_best[n] = (per[n]["F1"], conf)
    return {
        "best_mean_F1": best_mean[0],
        "best_mean_conf": best_mean[1],
        "best_per_class": {n: {"F1": f, "conf": c}
                           for n, (f, c) in per_class_best.items()},
        "at_best_mean": at[best_mean[1]] if best_mean[1] is not None else {},
    }


# ---------------------------------------------------------------------------
# contamination (paper metric: share of ALL GT folded into text-area)
# ---------------------------------------------------------------------------

def contamination(pages, conf: float, cover: float = 0.5):
    """Detect / absorb / clean-miss plus recoverable vs hidden split.

    For each GT clutter box (header, footer, footnote, header-footer):

    *same*  : greedy IoU>=0.5 match to a predicted box of that class.
    *in_ta* : >=`cover` of the GT area lies inside the predicted text-area
              envelope (the OCR crop). Checked whether or not *same*.
    *as_ta* : max IoU with any native predicted text-area box >= 0.5
              (boxes not merged). "Detected as text-area" under the same
              matching rule as *same*; TA boxes are not consumed.

    Partition of GT (envelope geometry, what actually hits the OCR crop):

      detected     : same
      recoverable  : same AND in_ta   (dual label; punch-out can fix)
      hidden       : not same AND in_ta  (= paper absorbed / contamination)
      clean-miss   : not same AND not in_ta

    `contamination_rate` is unchanged (hidden / all GT). The IoU-vs-TA
    rates (`as_textarea_*`) are a stricter "misclassified as text-area"
    diagnostic: a page-level TA envelope rarely hits IoU>=0.5 against a
    thin header, even when it covers it.
    """
    blank = dict(gt=0, det=0, absorbed=0, clean=0,
                 recoverable=0, hidden=0, in_ta=0,
                 as_ta=0, dual_as_ta=0, hidden_as_ta=0)
    stats = {k: dict(blank) for k in
             ("header-footer", "footnote", "header", "footer")}
    for stem, W, H, gboxes, pboxes in pages:
        g = [b for b in gboxes]  # native, conf ignored
        p = [b for b in pboxes if b["conf"] >= conf]
        gpx = [{**b, "xyxy": to_px(b, W, H)} for b in g]
        ppx = [{**b, "xyxy": to_px(b, W, H)} for b in p]
        ta = [b["xyxy"] for b in ppx if b["cls"] == 1]
        env = envelope_xyxy(ta) if ta else None

        def _one(gt_cls, pred_cls, key):
            gts = [b["xyxy"] for b in gpx if b["cls"] in gt_cls]
            prs = [b["xyxy"] for b in ppx if b["cls"] in pred_cls]
            used = [False] * len(prs)
            for gb in gts:
                stats[key]["gt"] += 1
                best, bj = 0.0, -1
                for j, pb in enumerate(prs):
                    if used[j]:
                        continue
                    v = iou_xyxy(gb, pb)
                    if v > best:
                        best, bj = v, j
                same = best >= 0.5 and bj >= 0
                if same:
                    used[bj] = True
                    stats[key]["det"] += 1
                covered = covered_frac(gb, env) >= cover
                as_ta = any(iou_xyxy(gb, tb) >= 0.5 for tb in ta)
                if covered:
                    stats[key]["in_ta"] += 1
                    if same:
                        stats[key]["recoverable"] += 1
                    else:
                        stats[key]["absorbed"] += 1
                        stats[key]["hidden"] += 1
                elif not same:
                    stats[key]["clean"] += 1
                if as_ta:
                    stats[key]["as_ta"] += 1
                    if same:
                        stats[key]["dual_as_ta"] += 1
                    else:
                        stats[key]["hidden_as_ta"] += 1

        _one((0, 3), (0, 3), "header-footer")
        _one((2,), (2,), "footnote")
        _one((0,), (0,), "header")
        _one((3,), (3,), "footer")

    out = {}
    for k, s in stats.items():
        n = s["gt"]
        def _r(c):
            return c / n if n else 0.0
        out[k] = {
            **s,
            "detected_rate": _r(s["det"]),
            "contamination_rate": _r(s["absorbed"]),
            "clean_miss_rate": _r(s["clean"]),
            "recoverable_contamination_rate": _r(s["recoverable"]),
            "hidden_contamination_rate": _r(s["hidden"]),
            "total_in_ta_rate": _r(s["in_ta"]),
            "as_textarea_rate": _r(s["as_ta"]),
            "dual_as_textarea_rate": _r(s["dual_as_ta"]),
            "hidden_as_textarea_rate": _r(s["hidden_as_ta"]),
        }
    return out


# ---------------------------------------------------------------------------
# Hidden Trespass (area-based) — the paper's primary text-area->clutter bleed.
# ---------------------------------------------------------------------------

def hidden_trespass(pages, conf: float, cover: float = 0.5):
    """Area-based text-area -> clutter bleed, micro-averaged over the test set.

    For each class c in {header-footer (0+3 combined), footnote (2)}:

        HT_c    = sum_pages area(E ∩ U_c) / sum_pages area(G_c)   (hidden)
        R_c     = sum_pages area(E ∩ D_c) / sum_pages area(G_c)   (removed)
        total_c = HT_c + R_c = sum area(E ∩ G_c) / sum area(G_c)

    where
        G_c = every GT region of class c (area summed over pages),
        E   = the predicted text-area envelope (min/max over predicted TA
              boxes on the page — the OCR crop; None if no TA predicted),
        U_c = class-c GT with NO class-c prediction at IoU>=0.5 (undetected,
              greedy matching — identical rule to contamination()'s `same`),
        D_c = G_c \\ U_c (detected).

    HT is COTe-Trespass restricted to MISSED regions: it is the continuous
    overlap area (no 50% threshold) between the OCR crop and the header /
    footer / footnote regions the model failed to also emit as their own
    class, so a punch-out post-process cannot recover them.

    The old count-based intuition is kept as a SECONDARY column:
        count_contamination_rate = (# class-c GT that are undetected AND have
        >=`cover` of their area inside E) / (# class-c GT).
    """
    keys = {"header-footer": (0, 3), "footnote": (2,)}
    acc = {k: dict(area_G=0.0, area_E_inter_U=0.0, area_E_inter_D=0.0,
                   n_gt=0, n_undetected=0, n_detected=0,
                   n_hidden_absorbed=0, n_pages_with_ta=0)
           for k in keys}
    for stem, W, H, gboxes, pboxes in pages:
        gpx = [{**b, "xyxy": to_px(b, W, H)} for b in gboxes]
        ppx = [{**b, "xyxy": to_px(b, W, H)} for b in pboxes if b["conf"] >= conf]
        ta = [b["xyxy"] for b in ppx if b["cls"] == 1]
        env = envelope_xyxy(ta) if ta else None
        for key, cls in keys.items():
            gts = [b["xyxy"] for b in gpx if b["cls"] in cls]
            prs = [b["xyxy"] for b in ppx if b["cls"] in cls]
            used = [False] * len(prs)
            for gb in gts:
                acc[key]["n_gt"] += 1
                acc[key]["area_G"] += box_area(gb)
                best, bj = 0.0, -1
                for j, pb in enumerate(prs):
                    if used[j]:
                        continue
                    v = iou_xyxy(gb, pb)
                    if v > best:
                        best, bj = v, j
                detected = best >= 0.5 and bj >= 0
                inter = inter_area(gb, env)
                if detected:
                    used[bj] = True
                    acc[key]["n_detected"] += 1
                    acc[key]["area_E_inter_D"] += inter
                else:
                    acc[key]["n_undetected"] += 1
                    acc[key]["area_E_inter_U"] += inter
                    if box_area(gb) > 0 and inter / box_area(gb) >= cover:
                        acc[key]["n_hidden_absorbed"] += 1
    out = {}
    for key, s in acc.items():
        g = s["area_G"] or 1.0
        n = s["n_gt"] or 1
        ht = s["area_E_inter_U"] / g
        r = s["area_E_inter_D"] / g
        out[key] = {
            "HT": ht,                       # hidden (undetected) bleed, area
            "R": r,                         # removed-before-OCR bleed, area
            "total_bleed": ht + r,          # HT + R
            "count_contamination_rate":     # SECONDARY intuition (old metric)
                s["n_hidden_absorbed"] / n,
            "n_gt": s["n_gt"],
            "n_undetected": s["n_undetected"],
            "n_detected": s["n_detected"],
            "n_hidden_absorbed": s["n_hidden_absorbed"],
            "area_G_px": s["area_G"],
            "area_E_inter_U_px": s["area_E_inter_U"],
            "area_E_inter_D_px": s["area_E_inter_D"],
        }
    return out


# ---------------------------------------------------------------------------
# Hidden Trespass v2 — exact POST-SUBTRACTION crop metric.
# ---------------------------------------------------------------------------

# Peripheral classes the production body crop actually paints out of the
# text-area crop canvas (bec-orchestration paddleocr_v2 `crop_body_regions`,
# blank_labels = footnote, header, footer). All non-body classes.
CROP_SUBTRACT_CLASSES = (0, 2, 3)  # header, footnote, footer


def hidden_trespass_crop(pages, conf: float,
                         subtract_classes=CROP_SUBTRACT_CLASSES):
    """Exact post-subtraction crop Hidden Trespass, micro-averaged over pages.

    For each page p, using predictions kept at >= `conf`:
      * E_p = smallest axis-aligned rectangle enclosing all retained predicted
              text-area boxes (None if none predicted);
      * P_p = geometric union of all retained peripheral prediction boxes whose
              classes are in `subtract_classes` (production subtracts header,
              footnote, footer from the body text-area crop);
      * C_p = E_p \\ P_p  is the actual OCR body crop.

    For peripheral GT class c in {header-footer (0+3), footnote (2)}:

        HT_c = sum_p sum_{g in G_{c,p}} area(C_p ∩ g)  /  sum_p sum_{g} area(g)

    i.e. the ground-truth peripheral AREA that still remains inside the
    post-subtraction body crop, over total GT area (the same denominator and
    micro-averaging convention as the previous metric).

    Diagnostic decomposition (exact identity, checked at runtime):
        HT_c = matched_residual_c + unmatched_residual_c
      where a GT box is *matched* iff it has a same-class prediction at
      IoU>=0.5 (greedy, the same rule as contamination()/operating_points()),
      *unmatched* otherwise. matched_residual is residual area from matched but
      incompletely removed regions; unmatched_residual is residual from
      unmatched (missed) GT.

    Also reported (all over the same denominator sum area(G_c)):
        raw_overlap_c = sum area(E_p ∩ g)          (pre-subtraction overlap)
        removed_c     = sum area(E_p ∩ g ∩ P_p)    (area removed by peripherals)
        raw_overlap_c = HT_c + removed_c            (identity)
    """
    keys = {"header-footer": (0, 3), "footnote": (2,)}
    acc = {k: dict(area_G=0.0, num_residual=0.0, num_raw=0.0, num_removed=0.0,
                   num_res_matched=0.0, num_res_unmatched=0.0,
                   n_gt=0, n_matched=0, n_unmatched=0, n_pages_no_ta=0)
           for k in keys}
    for stem, W, H, gboxes, pboxes in pages:
        gpx = [{**b, "xyxy": to_px(b, W, H)} for b in gboxes]
        ppx = [{**b, "xyxy": to_px(b, W, H)} for b in pboxes if b["conf"] >= conf]
        ta = [b["xyxy"] for b in ppx if b["cls"] == 1]
        env = envelope_xyxy(ta) if ta else None
        # P_p: union of all retained peripheral predictions to subtract.
        peri_rects = [b["xyxy"] for b in ppx if b["cls"] in subtract_classes]
        for key, cls in keys.items():
            gts = [b["xyxy"] for b in gpx if b["cls"] in cls]
            prs = [b["xyxy"] for b in ppx if b["cls"] in cls]
            used = [False] * len(prs)
            if env is None:
                acc[key]["n_pages_no_ta"] += 1
            for gb in gts:
                acc[key]["n_gt"] += 1
                acc[key]["area_G"] += box_area(gb)
                # matched status: greedy same-class IoU>=0.5
                best, bj = 0.0, -1
                for j, pb in enumerate(prs):
                    if used[j]:
                        continue
                    v = iou_xyxy(gb, pb)
                    if v > best:
                        best, bj = v, j
                matched = best >= 0.5 and bj >= 0
                if matched:
                    used[bj] = True
                e_inter_g, removed, residual = crop_residual(env, peri_rects, gb)
                acc[key]["num_raw"] += e_inter_g
                acc[key]["num_removed"] += removed
                acc[key]["num_residual"] += residual
                if matched:
                    acc[key]["n_matched"] += 1
                    acc[key]["num_res_matched"] += residual
                else:
                    acc[key]["n_unmatched"] += 1
                    acc[key]["num_res_unmatched"] += residual
    out = {}
    for key, s in acc.items():
        g = s["area_G"] or 1.0
        ht = s["num_residual"] / g
        mr = s["num_res_matched"] / g
        ur = s["num_res_unmatched"] / g
        out[key] = {
            "HT": ht,                              # post-crop hidden trespass
            "matched_residual": mr,                # from matched, incompletely removed
            "unmatched_residual": ur,              # from unmatched (missed) GT
            "raw_overlap": s["num_raw"] / g,       # pre-subtraction area(E∩G)/area(G)
            "removed": s["num_removed"] / g,       # area removed by peripheral preds
            "n_gt": s["n_gt"],
            "n_matched": s["n_matched"],
            "n_unmatched": s["n_unmatched"],
            "n_pages_no_ta": s["n_pages_no_ta"],
            "area_G_px": s["area_G"],
            "num_residual_px": s["num_residual"],
            "num_raw_px": s["num_raw"],
            "num_removed_px": s["num_removed"],
        }
    return out


# ---------------------------------------------------------------------------
# LED error types (arXiv 2603.17265): Missing / Merge / Split on clutter
# ---------------------------------------------------------------------------

def led_errors(pages, conf: float, classes="clutter"):
    """Count LED Missing / Merge / Split / Hallucination on header/footer/footnote.

    Definitions follow Heo et al. (LED):
      Missing: GT with no pred of the same class at IoU>=0.1
      Merge:   a pred with IoU>=0.1 against two or more distinct same-class GT
      Split:   one GT covered by n>=2 preds, each IoU<0.5 but sum IoU>=0.5
      Hallucination: pred with no GT of the same class at IoU>=0.1
    `classes` is 'clutter' (header, footer, footnote separately plus combined
    header-footer) or a list of native class ids.
    """
    keys = [("header", (0,)), ("footer", (3,)), ("footnote", (2,)),
            ("header-footer", (0, 3))]
    stats = {k: dict(gt=0, pred=0, missing=0, merge=0, split=0,
                     hallucination=0, matched=0)
             for k, _ in keys}
    for stem, W, H, gboxes, pboxes in pages:
        gpx = [{**b, "xyxy": to_px(b, W, H)} for b in gboxes]
        ppx = [{**b, "xyxy": to_px(b, W, H)} for b in pboxes if b["conf"] >= conf]
        for key, cls in keys:
            gts = [b["xyxy"] for b in gpx if b["cls"] in cls]
            prs = [b["xyxy"] for b in ppx if b["cls"] in cls]
            stats[key]["gt"] += len(gts)
            stats[key]["pred"] += len(prs)
            # Missing / matched
            for gb in gts:
                ious = [iou_xyxy(gb, pb) for pb in prs]
                if not ious or max(ious) < 0.1:
                    stats[key]["missing"] += 1
                else:
                    stats[key]["matched"] += 1
                # Split
                if len(ious) >= 2:
                    parts = [v for v in ious if v < 0.5]
                    if len(parts) >= 2 and sum(parts) >= 0.5:
                        stats[key]["split"] += 1
            # Merge / hallucination
            for pb in prs:
                hits = [j for j, gb in enumerate(gts) if iou_xyxy(gb, pb) >= 0.1]
                if len(hits) == 0:
                    stats[key]["hallucination"] += 1
                elif len(hits) >= 2:
                    stats[key]["merge"] += 1
    return stats


# ---------------------------------------------------------------------------
# COTe (cotescore library)
# ---------------------------------------------------------------------------

PERIPHERAL = ("header", "footer", "footnote")  # non-body COTe classes


def cote_page(W, H, gboxes, pboxes, conf: float, max_dim: int = 1024):
    """COTe on one page. Text-area envelope is the body SSU; each header /
    footer / footnote box is its own SSU. Predictions: one text-area envelope
    plus native header/footer/footnote boxes.

    Returns the scalar COTe decomposition (cote, coverage, overlap, trespass,
    excess) plus the class-resolved **text-area -> peripheral** trespass taken
    straight from cotescore's Tk,l confusion matrix (Eq. 15): how much of the
    predicted text-area (body) crop area trespasses onto header / footer /
    footnote GT SSUs. Returns None if the page has no GT SSU.
    """
    from cotescore import cote_score, trespass_matrix
    from cotescore.adapters import (
        boxes_to_gt_ssu_map, boxes_to_pred_masks, build_ssu_to_class, eval_shape,
    )
    from cotescore.types import MaskInstance

    gpx = [{**b, "xyxy": to_px(b, W, H)} for b in gboxes]
    ppx = [{**b, "xyxy": to_px(b, W, H)} for b in pboxes if b["conf"] >= conf]
    gt_boxes = []
    ssu = 1
    # text-area envelope first (semantic unit for the OCR crop)
    ta = [b["xyxy"] for b in gpx if b["cls"] == 1]
    if ta:
        e = envelope_xyxy(ta)
        gt_boxes.append({
            "x": e[0], "y": e[1], "width": e[2] - e[0], "height": e[3] - e[1],
            "ssu_id": ssu, "ssu_class": "text-area",
        })
        ssu += 1
    for b in gpx:
        if b["cls"] == 1:
            continue
        x1, y1, x2, y2 = b["xyxy"]
        cls_name = {0: "header", 2: "footnote", 3: "footer"}.get(b["cls"], "other")
        gt_boxes.append({
            "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1,
            "ssu_id": ssu, "ssu_class": cls_name,
        })
        ssu += 1
    if not gt_boxes:
        return None

    pred_boxes = []
    ta_p = [b["xyxy"] for b in ppx if b["cls"] == 1]
    if ta_p:
        e = envelope_xyxy(ta_p)
        pred_boxes.append({
            "x": e[0], "y": e[1], "width": e[2] - e[0], "height": e[3] - e[1],
            "class": "text-area",
        })
    for b in ppx:
        if b["cls"] == 1:
            continue
        x1, y1, x2, y2 = b["xyxy"]
        cls_name = {0: "header", 2: "footnote", 3: "footer"}.get(b["cls"], "other")
        pred_boxes.append({
            "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1,
            "class": cls_name,
        })

    cw, ch, scale = eval_shape(W, H, max_dim=max_dim)
    gt_map = boxes_to_gt_ssu_map(gt_boxes, cw, ch, scale=scale)
    ssu_to_class = build_ssu_to_class(gt_boxes)
    pmasks = boxes_to_pred_masks(pred_boxes, cw, ch, scale=scale)
    preds = [MaskInstance(mask=m, label=pb["class"])
             for m, pb in zip(pmasks, pred_boxes)]
    cote, C, O, T, E = cote_score(gt_map, preds)

    # class-resolved text-area -> peripheral trespass. T[k,l] is normalised by
    # A^P_k (class-k predicted area), so multiplying the peripheral row-sum by
    # A^P_{text-area} recovers the raw trespass pixel area for micro-averaging.
    ta_tp_frac = 0.0
    ta_pred_area = 0.0
    if preds:
        Tm, classes = trespass_matrix(gt_map, ssu_to_class, preds)
        if "text-area" in classes:
            k = classes.index("text-area")
            peri = [classes.index(c) for c in PERIPHERAL if c in classes]
            ta_tp_frac = float(sum(Tm[k, l] for l in peri))
        ta_masks = [m for m, pb in zip(pmasks, pred_boxes)
                    if pb["class"] == "text-area"]
        if ta_masks:
            ta_bin = np.zeros_like(ta_masks[0])
            for m in ta_masks:
                ta_bin |= m
            ta_pred_area = float(ta_bin.sum())
    return {
        "cote": float(cote), "coverage": float(C), "overlap": float(O),
        "trespass": float(T), "excess": float(E),
        "ta_trespass_peripheral_frac": ta_tp_frac,   # per-page, /A^P_ta
        "ta_pred_area": ta_pred_area,                # A^P_ta (px)
        "ta_trespass_peripheral_px": ta_tp_frac * ta_pred_area,
    }


def cote_dataset(pages, conf: float, max_dim: int = 1024):
    rows = []
    for stem, W, H, g, p in pages:
        r = cote_page(W, H, g, p, conf, max_dim=max_dim)
        if r is not None:
            rows.append(r)
    if not rows:
        return {k: float("nan") for k in
                ("cote", "coverage", "overlap", "trespass", "excess")}
    keys = ("cote", "coverage", "overlap", "trespass", "excess")
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    # class-resolved text-area -> peripheral trespass, two aggregations:
    #   macro = mean of the per-page fraction (parallels the scalar T above)
    #   micro = sum(trespass px) / sum(text-area pred px) — area-weighted,
    #           the like-for-like partner of the area-based Hidden Trespass.
    num = float(np.sum([r["ta_trespass_peripheral_px"] for r in rows]))
    den = float(np.sum([r["ta_pred_area"] for r in rows]))
    mean["ta_trespass_peripheral_macro"] = float(
        np.mean([r["ta_trespass_peripheral_frac"] for r in rows]))
    mean["ta_trespass_peripheral_micro"] = num / den if den else 0.0
    mean["ta_trespass_peripheral"] = mean["ta_trespass_peripheral_micro"]
    mean["n_pages"] = len(rows)
    return mean


# ---------------------------------------------------------------------------
# page iterator
# ---------------------------------------------------------------------------

def iter_pages(gt_dir: Path, pred_dir: Path, sizes: dict, conf_floor: float = 0.0):
    stems = sorted(p.stem for p in gt_dir.glob("*.txt"))
    for stem in stems:
        if stem not in sizes:
            continue
        W, H = sizes[stem]
        gt = read_yolo(gt_dir / f"{stem}.txt", conf_floor=0.0)
        pred = read_yolo(pred_dir / f"{stem}.txt", conf_floor=conf_floor)
        yield stem, W, H, gt, pred
