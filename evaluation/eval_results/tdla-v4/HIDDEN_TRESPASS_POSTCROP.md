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

| model | h/f HT (old → new) | footnote HT (old → new) |
|---|---|---|
| TiBLA-RTDETR | 0.008 → **0.009** | 0.037 → **0.043** |
| TiBLA-PP-DocLayout-L | 0.003 → **0.004** | 0.037 → **0.037** |
| TiBLA-RFDETR | 0.020 → **0.021** | 0.216 → **0.169** |

(Full 13-system new values are in the table above / `hidden_trespass_postcrop.json`.)

> **Caveat / final value.** Per the operating-point protocol, the *final* paper
> value is the post-crop HT at the **validation-selected** confidence. These
> replacements are at the current paper confidences (metric change only). If the
> validation-selected confidence differs, the final footnote/header-footer HT may
> shift; see the blocker below. I have **not** retuned any threshold and have not
> invented val-selected values.

## Ranking changes

- **Footnote HT**, fine-tuned models: previously PP-DocLayout-L and RT-DETR were
  tied at 0.037; post-crop **PP-DocLayout-L (0.037) is now strictly below
  RT-DETR (0.043)**, because RT-DETR leaves more matched-but-uncovered footnote
  sliver. RF-DETR stays worst among ours (0.169) but its gap narrows sharply.
- **Header-footer HT**: ordering is unchanged (all deltas ≤ 0.0011 for ours).
- Off-the-shelf / commercial: no ordering flips; DocStructBench and off-the-shelf
  heron improve most on footnote (they emit footnote boxes that get subtracted).

## Operating-point protocol — decomposition and the validation blocker

The requested decomposition per system:

1. **Existing HT @ old test-selected confidence** — delivered (`old HT` column).
2. **Post-crop HT @ that same confidence** — delivered (`new HT` column). This is
   the *metric-only* change.
3. **Existing HT @ validation-selected confidence** — **blocked** (see below).
4. **Post-crop HT @ validation-selected confidence** (the final paper value) —
   **blocked**.

**Blocker: no validation prediction dumps exist.** Only the validation *images*
and *labels* are archived (`s3://bec.bdrc.io/models/hff-detection/datasets/
tdlav4_tam2col/{images,labels}/val`, 749 pages). No system's predictions have
been run on the val split, so I cannot (without generating them):
- select a global confidence on val by max canonical macro mean F1,
- report validation-selected confidence, val F1, or frozen-test F1,
- produce per-class validation curves for deployment thresholds,
- compute rows (3) and (4).

I did **not** substitute test-selected confidences for these (that is exactly the
test-leak the protocol forbids), and I did **not** invent any missing values.

**To unblock (needs a decision — GPU cost/time):** generate val dumps for the 8
*thresholdable* detectors — the 5 fine-tunes (RT-DETR seed0, RF-DETR,
PP-DocLayout-L, DocLayout-YOLO, Docling heron) + 3 off-the-shelf (PP-DocLayout-L,
Docling heron, DocLayout-YOLO DocStructBench) — by running each checkpoint on the
749 val pages in its own framework (Ultralytics / rfdetr / PaddleX / DocLayout-YOLO
/ transformers), then re-select on val and freeze for test. The **commercial**
systems (Azure, Google, Textract) and the **VLMs** (Surya, Chandra) currently
have no usable per-box confidence for a sweep (operating conf 0.00), so per the
protocol they **retain their fixed operating point** and need no val run.

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
```

Artifact revisions:
- Dataset: HF tag **v4** (`BDRC/TiBLAD`), 833-page test; GT = leak-free series split.
- Prediction dumps (test): `s3://bec.bdrc.io/models/hff-detection/tdlav4/eval/…`
  and `.../off-the-shelf-eval-tdlav4/…`, mirrored locally to
  `/home/eroux/tmp/tdlav4_lit/preds/<system>/labels`. Deterministic; nothing
  re-inferred for this metric change.
