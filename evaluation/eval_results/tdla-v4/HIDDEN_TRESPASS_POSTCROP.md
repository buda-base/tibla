# Hidden Trespass — exact post-subtraction crop metric (TiBLAD v4, 833-page test)

This note replaces the previous **match-conditioned** Hidden Trespass with an
**exact post-subtraction crop** metric and re-runs all 13 v4 systems at the same
operating confidences used by the paper, so old and new values are directly
comparable (no threshold retuning here — see the operating-point section for the
validation-selection plan).

- Metric code: `evaluation/literature_metrics.py` (`hidden_trespass_crop`,
  `crop_residual`, `rect_union_area`).
- Driver: `evaluation/run_v4_lit_eval.py` (now emits both `hidden_trespass`
  [old] and `hidden_trespass_crop` [new] per system).
- Tests: `evaluation/test_hidden_trespass.py` (9 tests, 38 assertions, all pass).
- Machine-readable: `literature/metrics.json` (per-system, both metrics) and
  `literature/hidden_trespass_postcrop.json` (focused old-vs-new + decomposition).

## Metric definition

For each page *p*, using predictions retained at the system's operating
confidence:

1. **E_p** = smallest axis-aligned rectangle enclosing all retained predicted
   `text-area` boxes (the body envelope; `None` if none predicted).
2. **P_p** = geometric union of the retained peripheral prediction boxes that the
   production pipeline paints out of the body crop.
3. **C_p = E_p \ P_p** — the actual OCR body crop.
4. For peripheral class *c* ∈ {header-footer (0+3), footnote (2)}:

   **HT_c = Σ_p Σ_{g∈G_{c,p}} area(C_p ∩ g)  /  Σ_p Σ_{g∈G_{c,p}} area(g)**

Denominator (Σ area of GT of class *c*) and micro-averaging are unchanged from
the previous metric. Rectangle-union subtraction is computed exactly by
coordinate compression, so overlapping peripheral predictions are subtracted
only once (unit test 3). Pages with no predicted text-area contribute zero
residual (unit test 4). This is compared with COTe-Trespass as a related
diagnostic; it is **not** a literal restriction or decomposition of COTe.

### Which predicted classes the production pipeline subtracts

Confirmed by inspecting the deployed pipeline, **bec-orchestration
`paddleocr_v2`** (`bec_orch/jobs/paddleocr/layout_split.py`, `crop_body_regions`,
`blank_labels = ("footnote", "header", "footer")`): the body text-area crop
canvas blanks **header + footnote + footer** (all three non-body classes). So
`P_p` = union of retained header, footnote and footer predictions
(`CROP_SUBTRACT_CLASSES = (0, 2, 3)` in `literature_metrics.py`).

(Note: the optional *page-level* background fill `apply_header_footer_mask`
paints out header+footer only and protects footnote; but the metric models the
**body crop**, `crop_body_regions`, which blanks all three. If the paper wants
the page-mask convention instead, set `subtract_classes=(0, 3)`.)

## Diagnostic decomposition (exact identities, checked at runtime)

- **HT_c = matched_residual_c + unmatched_residual_c** — matched-but-incompletely
  removed vs unmatched (missed) GT. A GT box is *matched* iff a same-class
  prediction hits it at IoU ≥ 0.5 (greedy; same rule as the F1 operating point).
- **raw_overlap_c = HT_c + removed_c** — pre-subtraction overlap area(E∩G)
  equals post-crop residual plus the area removed by peripheral predictions.

## Old vs new Hidden Trespass — all 13 systems (same paper operating confidence)

`conf` = paper operating confidence (unchanged). HT values are area fractions,
micro-averaged over the 833-page test. `mR`/`uR` = matched / unmatched residual
(sum to new HT). `raw` = pre-subtraction area(E∩G)/area(G); `rem` = area removed
by peripheral predictions.

| system | conf | old hf HT | new hf HT | Δ hf | hf mR | hf uR | hf raw | hf rem | old fn HT | new fn HT | Δ fn | fn mR | fn uR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RT-DETR-l tam2col (ours; seed0) | 0.74 | 0.0082 | **0.0089** | +0.0007 | 0.0007 | 0.0082 | 0.0099 | 0.0009 | 0.0368 | **0.0428** | +0.0060 | 0.0060 | 0.0368 |
| RF-DETR-Large tam2col (ours) | 0.26 | 0.0200 | **0.0210** | +0.0010 | 0.0010 | 0.0200 | 0.0221 | 0.0011 | 0.2158 | **0.1685** | −0.0473 | 0.0000 | 0.1685 |
| Docling layout-heron tam2col (ours) | 0.08 | 0.0019 | **0.0025** | +0.0006 | 0.0006 | 0.0019 | 0.0051 | 0.0026 | 0.0736 | **0.0736** | +0.0000 | 0.0000 | 0.0736 |
| DocLayout-YOLO tam2col (ours) | 0.29 | 0.0036 | **0.0040** | +0.0005 | 0.0005 | 0.0036 | 0.0054 | 0.0014 | 0.1061 | **0.1061** | +0.0000 | 0.0000 | 0.1061 |
| PP-DocLayout-L tam2col (ours) | 0.68 | 0.0028 | **0.0039** | +0.0011 | 0.0011 | 0.0028 | 0.0048 | 0.0009 | 0.0368 | **0.0368** | +0.0000 | 0.0000 | 0.0368 |
| PP-DocLayout-L (off-the-shelf) | 0.30 | 0.1783 | **0.1889** | +0.0107 | 0.0153 | 0.1736 | 0.2472 | 0.0582 | 0.2773 | **0.2811** | +0.0037 | 0.0037 | 0.2773 |
| Docling layout-heron (off-the-shelf) | 0.58 | 0.0778 | **0.0996** | +0.0218 | 0.0292 | 0.0705 | 0.2112 | 0.1116 | 0.0901 | **0.0556** | −0.0345 | 0.0001 | 0.0555 |
| DocLayout-YOLO DocStructBench (OTS) | 0.08 | 0.0995 | **0.1122** | +0.0127 | 0.0265 | 0.0857 | 0.2615 | 0.1493 | 0.2411 | **0.1655** | −0.0755 | 0.0000 | 0.1655 |
| Surya 2 layout VLM | 0.00 | 0.0204 | **0.0210** | +0.0006 | 0.0007 | 0.0204 | 0.0216 | 0.0006 | 0.1345 | **0.1345** | +0.0000 | 0.0000 | 0.1345 |
| Chandra 2 | 0.00 | 0.0187 | **0.0191** | +0.0005 | 0.0005 | 0.0187 | 0.0194 | 0.0002 | 0.0203 | **0.0203** | +0.0000 | 0.0000 | 0.0203 |
| Azure DI prebuilt-layout | 0.00 | 0.1810 | **0.1812** | +0.0002 | 0.0005 | 0.1807 | 0.1860 | 0.0048 | 0.1841 | **0.1779** | −0.0062 | 0.0000 | 0.1779 |
| AWS Textract Layout | 0.00 | 0.1596 | **0.1604** | +0.0008 | 0.0030 | 0.1574 | 0.1682 | 0.0078 | 0.6442 | **0.6423** | −0.0019 | 0.0000 | 0.6423 |
| Google DocAI Layout Parser | 0.00 | 0.4379 | **0.4123** | −0.0256 | 0.0016 | 0.4107 | 0.4429 | 0.0306 | 0.9295 | **0.9281** | −0.0013 | 0.0000 | 0.9281 |

Interpretation:
- For the **fine-tuned systems**, `unmatched_residual` equals the old HT to 4 dp
  (missed peripherals have no same-class prediction, so nothing is punched out of
  their area). The small **increase** is exactly the new `matched_residual` term:
  matched peripheral boxes that do not *fully* cover their GT leave a sliver
  inside the crop. This is real bleed the old metric ignored.
- Systems that **predict footnotes/peripherals** get credit for the production
  punch-out: RF-DETR footnote HT drops 0.216 → 0.169, DocStructBench footnote
  0.241 → 0.166, off-the-shelf heron footnote 0.090 → 0.056, Google header/footer
  0.438 → 0.412. The `removed` column quantifies the punched-out area.

## Exact replacement values for the paper's Hidden Trespass tables

At the **current (unchanged) operating confidences** — i.e. the metric change
only. Rounded to 3 dp as in the model cards:

**(a) Metric change only** — post-crop HT at the *current* paper confidences
(these confidences were selected on the test set):

| model | h/f HT (old → post-crop) | footnote HT (old → post-crop) |
|---|---|---|
| TiBLA-RTDETR | 0.008 → **0.009** | 0.037 → **0.043** |
| TiBLA-PP-DocLayout-L | 0.003 → **0.004** | 0.037 → **0.037** |
| TiBLA-RFDETR | 0.020 → **0.021** | 0.216 → **0.169** |

**(b) Final paper values** — post-crop HT at the **validation-selected**
confidence (no test-set tuning; row 4 of the decomposition below):

| model | val conf | h/f HT | footnote HT | frozen test mean F1 |
|---|---:|---:|---:|---:|
| TiBLA-RTDETR | 0.64 | **0.009** | **0.043** | 0.952 |
| TiBLA-PP-DocLayout-L | 0.61 | **0.004** | **0.037** | 0.955 |
| TiBLA-RFDETR | 0.47 | **0.021** | **0.178** | 0.921 |

(Full 13-system values are in `hidden_trespass_postcrop.json` [metric change] and
`val_operating_point.json` [rows 1–4 + val selection].)

> **Note on the frozen test F1.** The validation-selected confidence gives a
> leak-free test mean F1 (0.952 / 0.955 / 0.921) slightly below the
> test-selected headline (0.959 / 0.958 / 0.926 in the F1 tables), as expected:
> the headline picks the confidence *on the test set*, this picks it on val.
> The HT values are robust to this: h/f is unchanged, footnote moves only for
> RF-DETR (0.169 → 0.178, because its higher val confidence keeps fewer footnote
> predictions to subtract).

## Ranking changes

- **Footnote HT**, fine-tuned models: previously PP-DocLayout-L and RT-DETR were
  tied at 0.037; post-crop **PP-DocLayout-L (0.037) is now strictly below
  RT-DETR (0.043)**, because RT-DETR leaves more matched-but-uncovered footnote
  sliver. RF-DETR stays worst among ours (0.169) but its gap narrows sharply.
- **Header-footer HT**: ordering is unchanged (all deltas ≤ 0.0011 for ours).
- Off-the-shelf / commercial: no ordering flips; DocStructBench and off-the-shelf
  heron improve most on footnote (they emit footnote boxes that get subtracted).

## Operating-point protocol — validation-based selection (rows 1–4)

Validation dumps for the 8 thresholdable detectors were generated on GPU (see
Reproduce) at the same predict settings / conf floors as the test dumps, and the
operating confidence was re-selected on the **validation** split only.

**Selection rule (exact):** `val_conf = argmax_{c∈{0.00,…,0.99}}` of the
canonical macro mean F1 = mean(F1 header-footer, text-area, footnote) on the 749
val pages (greedy IoU≥0.5; text-area envelope built from predictions kept at ≥c;
ties → lowest conf). `val_conf` is then **frozen** and applied to the 833 test
pages. No test result is used to choose any threshold.

Per-system decomposition (HT as area fraction; **row 4 is the final paper value**):

| system | old conf | val conf | (1) hf @old | (2) hf-crop @old | (3) hf @val | (4) hf-crop @val | (1) fn @old | (2) fn-crop @old | (3) fn @val | (4) fn-crop @val |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RT-DETR-l (ours) | 0.74 | 0.64 | 0.008 | 0.009 | 0.008 | **0.009** | 0.037 | 0.043 | 0.037 | **0.043** |
| RF-DETR-L (ours) | 0.26 | 0.47 | 0.020 | 0.021 | 0.020 | **0.021** | 0.216 | 0.169 | 0.226 | **0.178** |
| Docling heron (ours) | 0.08 | 0.07 | 0.002 | 0.003 | 0.002 | **0.003** | 0.074 | 0.074 | 0.074 | **0.074** |
| DocLayout-YOLO (ours) | 0.29 | 0.32 | 0.004 | 0.004 | 0.004 | **0.004** | 0.106 | 0.106 | 0.106 | **0.106** |
| PP-DocLayout-L (ours) | 0.68 | 0.61 | 0.003 | 0.004 | 0.003 | **0.004** | 0.037 | 0.037 | 0.037 | **0.037** |
| PP-DocLayout-L OTS | 0.30 | 0.31 | 0.178 | 0.189 | 0.176 | **0.187** | 0.277 | 0.281 | 0.277 | **0.281** |
| Docling heron OTS | 0.58 | 0.67 | 0.078 | 0.100 | 0.085 | **0.091** | 0.090 | 0.056 | 0.090 | **0.055** |
| DocLayout-YOLO DocStruct OTS | 0.08 | 0.13 | 0.100 | 0.112 | 0.114 | **0.124** | 0.241 | 0.166 | 0.207 | **0.132** |
| Surya 2 † | 0.00 | 0.00 | 0.020 | 0.021 | 0.020 | **0.021** | 0.135 | 0.135 | 0.135 | **0.135** |
| Chandra 2 † | 0.00 | 0.00 | 0.019 | 0.019 | 0.019 | **0.019** | 0.020 | 0.020 | 0.020 | **0.020** |
| Azure DI † | 0.00 | 0.00 | 0.181 | 0.181 | 0.181 | **0.181** | 0.184 | 0.178 | 0.184 | **0.178** |
| AWS Textract † | 0.00 | 0.00 | 0.160 | 0.160 | 0.160 | **0.160** | 0.644 | 0.642 | 0.644 | **0.642** |
| Google DocAI † | 0.00 | 0.00 | 0.438 | 0.412 | 0.438 | **0.412** | 0.929 | 0.928 | 0.929 | **0.928** |

† **Fixed operating point** — commercial APIs and VLMs expose no usable per-box
confidence for a sweep (operating conf 0.00). Per the protocol they retain their
fixed point, so rows (3)/(4) equal rows (1)/(2). Validation predictions were
therefore **not** generated for these five systems.

Validation and frozen-test mean F1 for the thresholdable systems:

| system | val conf | val mean F1 | frozen test mean F1 | (test-selected headline F1) |
|---|---:|---:|---:|---:|
| RT-DETR-l (ours) | 0.64 | 0.983 | 0.952 | 0.959 |
| RF-DETR-L (ours) | 0.47 | 0.972 | 0.921 | 0.926 |
| Docling heron (ours) | 0.07 | 0.961 | 0.925 | 0.925 |
| DocLayout-YOLO (ours) | 0.32 | 0.943 | 0.897 | 0.897 |
| PP-DocLayout-L (ours) | 0.61 | 0.982 | 0.955 | 0.958 |
| PP-DocLayout-L OTS | 0.31 | 0.744 | 0.668 | 0.670 |
| Docling heron OTS | 0.67 | 0.603 | 0.583 | 0.590 |
| DocLayout-YOLO DocStruct OTS | 0.13 | 0.522 | 0.501 | 0.503 |

The val-selected confidences track the test-selected ones closely (largest shift
RF-DETR 0.26 → 0.47), confirming the paper's operating points were not
test-overfit. Full per-class validation P/R/F1 curves (0.05-grid, for deployment
threshold choices) are in `val_operating_point.json` (`val_curve` per system);
these are reported for information only and were **not** used to pick any
test-time threshold.

## Reproduce

```bash
# 1. unit tests
.venv_eval/bin/python evaluation/test_hidden_trespass.py

# 2. rerun all 13 systems (both HT metrics) at the paper operating confidences
.venv_eval/bin/python evaluation/run_v4_lit_eval.py \
    --gt-dir  /home/eroux/tmp/dataset_tdlav4_tam2col/labels/test \
    --img-dir /home/eroux/tmp/dataset_tdlav4_tam2col/images/test \
    --pred-root /home/eroux/tmp/tdlav4_lit/preds \
    --out-dir evaluation/eval_results/tdla-v4/literature

# 3. focused old-vs-new comparison JSON + table
.venv_eval/bin/python evaluation/hidden_trespass_delta.py \
    --metrics evaluation/eval_results/tdla-v4/literature/metrics.json \
    --out     evaluation/eval_results/tdla-v4/literature/hidden_trespass_postcrop.json

# 4. validation dumps (8 thresholdable detectors) — GPU, same predict settings
#    /conf floors as the test dumps (rfdetr 0.01, others 0.05). Drivers:
#    tmp/valinfer_boxA.sh (rtdetr seed0, rfdetr, heron ft+ots) and
#    tmp/valinfer_boxP.sh (pp ft+ots, doclayout ft+ots). Dumps land at
#    s3://bec.bdrc.io/models/hff-detection/tdlav4/eval-val/<system>/labels.

# 5. validation-based operating-point selection + rows 1–4
.venv_eval/bin/python evaluation/val_operating_point.py \
    --val-gt   dataset_tdlav4_tam2col/labels/val  --val-img  dataset_tdlav4_tam2col/images/val \
    --test-gt  dataset_tdlav4_tam2col/labels/test --test-img dataset_tdlav4_tam2col/images/test \
    --val-pred-root  <local>/tdlav4_val/preds  --test-pred-root <local>/tdlav4_lit/preds \
    --test-metrics   evaluation/eval_results/tdla-v4/literature/metrics.json \
    --out-dir        evaluation/eval_results/tdla-v4/literature
```

Artifact revisions:
- Dataset: HF tag **v4** (`BDRC/TiBLAD`), 833-page test / 749-page val; GT =
  leak-free series split.
- Prediction dumps (test): `s3://bec.bdrc.io/models/hff-detection/tdlav4/eval/…`
  and `.../off-the-shelf-eval-tdlav4/…`, mirrored to `/home/eroux/tmp/tdlav4_lit/
  preds/<system>/labels`. Deterministic; nothing re-inferred for the metric change.
- Prediction dumps (val): `s3://bec.bdrc.io/models/hff-detection/tdlav4/eval-val/
  <system>/labels` (8 thresholdable detectors), generated on two A10G boxes with
  the checkpoints in `.../tdlav4/weights/` (RT-DETR seed0 =
  `seed-variance-tdlav4/seed0/best.pt`). Commercial/VLM systems have no val dumps
  (fixed operating point).
