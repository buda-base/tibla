# TiBLA v4 — commercial-API evaluation provenance manifest

Stable, verifiable provenance for the three commercial document-AI baselines
reported in the TiBLA paper (`2026-tibetan-book-layout`): **Azure AI Document
Intelligence `prebuilt-layout`**, **AWS Textract `AnalyzeDocument` (`LAYOUT`)**,
and **Google Cloud Document AI Layout Parser**.

> **Status.** These evaluations are **auditable and deterministically
> re-scorable from the archived derived detections**. They are **not fully
> reproducible**: the raw API responses were **not** retained, so the
> native-label mappings, polygons, reading order, text, and provider response
> metadata cannot be re-derived without re-calling the services (which are
> managed and, for Azure/Textract, versionless — see *Model mutability*).

## Dataset

| field | value |
|---|---|
| Dataset | `BDRC/TiBLAD` |
| Hub tag | `v4` |
| Split | `test` (series-level, leakage-free) |
| Pages | **833** |
| Image MIME | `image/jpeg` (JFIF baseline, 3-channel) |

## Run

| field | value |
|---|---|
| Run date | **2026-08-29** (≈11:38–12:04 UTC) |
| Prediction-script commit | **`ce3a77e`** |
| Runtime | Python 3.11.2 |
| SDKs (best-effort) | boto3 1.37.24 · botocore 1.37.38 · requests 2.32.3 · google-cloud-documentai 3.15.0 · google-api-core 2.29.0 · pillow 12.1.1 |
| Batching | `ThreadPoolExecutor`, 8 workers, one page per synchronous request |
| Outcome | 833 submitted / 833 analyzed / 0 failed / 0 skipped / 0 duplicated (all three services) |
| Raw API responses retained | **No** |

## Service configurations

### Azure AI Document Intelligence — `prebuilt-layout`
- Operation: `POST …/documentintelligence/documentModels/prebuilt-layout:analyze`
  (async, polled via `Operation-Location`).
- **API version `2024-11-30`**. Region: **UNKNOWN** (endpoint held in an
  environment variable; not recorded).
- Input: **raw JPEG bytes** (`application/octet-stream`), no preprocessing.
- Per-box confidence: **none** → single fixed, un-thresholdable operating point
  (score := 1.0).
- Footnote class: **present** (only commercial provider with a `footnote` role).

### AWS Textract — `AnalyzeDocument` `LAYOUT`
- Operation: `AnalyzeDocument`, `FeatureTypes=["LAYOUT"]`, region **`us-east-1`**.
- API/model version: **UNKNOWN** (versionless managed service).
- Input: **raw JPEG bytes**, no preprocessing.
- Per-box confidence: **yes** (`Confidence/100`, 6th label column). Operating
  point is nonetheless **fixed at confidence 0.00** because no Textract
  *validation* predictions were generated (no validation-selected point).
  Coincidentally the test best-mean-F1 confidence is also 0.00, so the fixed
  point equals the test optimum; no test-set tuning was used.
- Footnote class: **absent** → footnote F1 = 0 by construction.

### Google Cloud Document AI — Layout Parser
- Operation: `process_document` on `us-documentai.googleapis.com`.
- **Processor version `pretrained-layout-parser-v1.0-2024-06-03`** (stable,
  explicitly pinned), location **`us`**.
- Input: **single-page PDF generated from each JPEG** (PIL
  `convert('RGB').save(format='PDF')`), because Layout Parser rejects
  `image/*`. This preprocessing step is **not** applied to Azure/Textract.
- Per-box confidence: **none** → single fixed, un-thresholdable operating point
  (score := 1.0).
- Footnote class: **absent** → footnote F1 = 0 by construction.

## Native → TiBLAD class mappings

| provider | header | footer | footnote | text-area |
|---|---|---|---|---|
| Azure `prebuilt-layout` | `pageHeader` | `pageFooter`, `pageNumber` | `footnote` | everything else (title, sectionHeading, None, …) |
| AWS Textract `LAYOUT` | `LAYOUT_HEADER` | `LAYOUT_FOOTER`, `LAYOUT_PAGE_NUMBER` | *(absent)* | other `LAYOUT_*`; non-`LAYOUT_*` dropped |
| Google Layout Parser | `header` | `footer` | *(absent)* | paragraph, subtitle, heading-1..5, list, table, image |

The scorer then folds these four classes into the canonical three: `{header,
footer}` → **header-footer** (matched individually), **text-area** → one page/
column envelope, **footnote** unchanged; matches at IoU ≥ 0.5.

## Results (TiBLAD v4, 833-page test)

| system | hf F1 | ta F1 | fn F1 | mean F1 | shared mAP@.5:.95 | HT-crop h/f | HT-crop fn |
|---|---:|---:|---:|---:|---:|---:|---:|
| Azure `prebuilt-layout` | 0.611 | 0.990 | 0.252 | 0.618 | 0.103 | 18.1% | 17.8% |
| AWS Textract `LAYOUT` | 0.232 | 0.835 | 0.000 | 0.356 | 0.004 | 16.0% | 64.2% |
| Google Layout Parser | 0.151 | 0.967 | 0.000 | 0.373 | 0.006 | 41.2% | 92.8% |

All values re-derive exactly from the archived detections (see *Verification*).

## Derived detections in this repository

The derived detections are **committed here** so the manifest is self-contained
and verifiable with no credentials. The payload is small (~267 KB total), so it
is stored as reproducible per-provider gzip tarballs (not Git LFS/Xet):

```
detections/azure_di.labels.tar.gz       # labels/<stem>.txt  × 833
detections/aws_textract.labels.tar.gz   # labels/<stem>.txt  × 833
detections/google_docai.labels.tar.gz   # labels/<stem>.txt  × 833
detections/SHA256SUMS                    # SHA-256 of the three tarballs
```

Each tarball extracts to `labels/<stem>.txt`. The tarballs are byte-reproducible
(`tar --sort=name --mtime='2026-08-29 00:00:00 UTC' --owner=0 --group=0
--numeric-owner … | gzip -n -9`); their SHA-256 are in `detections/SHA256SUMS`
and `provenance.json`. The authoritative reference remains the per-file
manifests under `sha256/` — `verify.sh` extracts the tarballs and checks every
file against them.

### Original archive (S3)

The same detections were originally archived on S3:

```
s3://bec.bdrc.io/models/hff-detection/tdlav4/eval/azure_di/labels
s3://bec.bdrc.io/models/hff-detection/tdlav4/eval/aws_textract/labels
s3://bec.bdrc.io/models/hff-detection/tdlav4/eval/gdocai/labels    # canonical id: google_docai
```
Each prefix holds 833 label `.txt` files plus `best.txt` and `data.yaml`.

**Access conditions.** These S3 objects are **not anonymously accessible**:
anonymous listing returns `AccessDenied` and anonymous HTTPS GET returns HTTP
`403` (checked 2026-09-07); **authenticated AWS access is required**. The in-repo
tarballs above are the credential-free equivalent.

## What is and is not reproducible

- **Deterministic re-scoring from derived detections — yes.** Given the archived
  four-class YOLO label files and the frozen v4 test images/ground truth, the
  scorer (`run_v4_lit_eval.py` + `literature_metrics.py`) reproduces every paper
  metric bit-for-bit. Greedy IoU matching and pycocotools are deterministic.
- **Re-deriving native mappings or provider metadata from raw responses — no.**
  Raw API responses were not stored. Only the derived 4-class boxes (and, for
  Textract, per-box confidence) survive. Alternative label maps, original
  polygons, reading order, text, request IDs, or served model versions cannot be
  recovered without re-calling the services.

### Model mutability
Azure `prebuilt-layout` (api-version `2024-11-30`) pins the API contract but the
underlying model may be updated by the provider under the same version; AWS
Textract is a versionless managed service that may change at any time. Served
model versions for both are **UNKNOWN and mutable**. Re-calling either service
may therefore return different results. Google's processor version is pinned.

## Manifests

Portable per-file SHA-256 manifests under `sha256/` use **relative basenames
only** (no absolute paths), sorted **bytewise (`LC_ALL=C`)** by filename. The
**aggregate** for each is the SHA-256 of the sorted manifest file:

| manifest | files | aggregate SHA-256 |
|---|---:|---|
| `sha256/images.sha256` | 833 | `2b1676c4c0d80641b23116c0ecac8f40cee80842ebe5d5b7d160660e080dceba` |
| `sha256/gt_labels.sha256` | 833 | `27c40996bdb5090d661961fb1763dab0f98dff6a911a91e716992cea1504c3d9` |
| `sha256/azure_di.sha256` | 833 | `3497c147826333aeb1503e29409bfabdc9bc949fa75e68e8167c67f36dfb56bc` |
| `sha256/aws_textract.sha256` | 833 | `8272fb903dd6c15a7122d8c59c6f13ae016f251c6ee6ec95db24ee401e151e79` |
| `sha256/google_docai.sha256` | 833 | `4af59abbd9b77bd605d1917feb0db155beb76557dc79d08ff90b9d767c417372` |

All five manifests share an identical set of 833 page stems.

## Download & verify

**Self-contained (no credentials, no network)** — verify the in-repo detections:

```bash
./verify.sh --detections ./detections
```

This checksum-verifies the three tarballs against `detections/SHA256SUMS`,
extracts them, checks every per-file SHA-256 against `sha256/`, recomputes the
portable aggregates, checks the 833 counts and stem alignment, and validates
`provenance.json`.

**Optional — also verify the gated test set.** The v4 test images and
ground-truth labels are distributed via the gated dataset `BDRC/TiBLAD`
(tag `v4`); point the extra flags at your local copy:

```bash
./verify.sh --detections ./detections \
    --images <v4-test>/images/test \
    --gt     <v4-test>/labels/test
```

Explicit `--azure/--textract/--google DIR` flags override the tarball payload if
you have the detections extracted elsewhere (e.g. synced from S3 with
authenticated AWS access).

## Exact re-scoring command

```bash
# extract the in-repo detections into a pred-root layout
mkdir -p /tmp/preds
for id in azure_di aws_textract google_docai; do
  mkdir -p /tmp/preds/$id
  tar -xzf detections/$id.labels.tar.gz -C /tmp/preds/$id   # -> /tmp/preds/$id/labels
done

# from the repository root, with the gated v4 test set
.venv_eval/bin/python evaluation/run_v4_lit_eval.py \
    --gt-dir  <v4-test>/labels/test \
    --img-dir <v4-test>/images/test \
    --pred-root /tmp/preds \             # azure_di/ aws_textract/ google_docai/, each with labels/
    --out-dir /tmp/tdla-v4-commercial-rescore \
    --systems azure_di,aws_textract,google_docai
```

This reproduces the mean F1, per-class F1, and post-subtraction Hidden-Trespass
values in the table above at each system's documented fixed operating point
(confidence 0.00).
