#!/usr/bin/env bash
# Val-split inference on box A (A10G): rtdetr seed0, rfdetr, docling heron ft+ots.
# Same predict settings / conf floors as the v4 TEST dumps (rtdetr/heron 0.05,
# rfdetr 0.01). Writes YOLO labels "cls cx cy w h conf" and syncs to S3.
set -uo pipefail
export HOME=/home/ubuntu
cd /home/ubuntu/tdlav4
S3=s3://bec.bdrc.io/models/hff-detection/tdlav4
VS3=$S3/eval-val
IMG=dataset_tdlav4_tam2col/images/val
PY=/opt/pytorch/bin/python
RF=/home/ubuntu/tdlav4/.venv_rfdetr/bin/python
mkdir -p eval-val weights_val
LOG=/home/ubuntu/tdlav4/valA.log
echo "=== $(date -u) BOXA val infer start ($(ls $IMG|wc -l) val images) ===" | tee -a $LOG

# 1) RT-DETR-l seed0 (ultralytics, conf floor 0.05, imgsz 1024)
S=rtdetr_tdlav4
if [ ! -f eval-val/$S/labels/.done ]; then
  aws s3 cp $S3/weights/rtdetr_tdlav4_tam2col_seed0_best.pt weights_val/rtdetr_seed0.pt --quiet
  mkdir -p eval-val/$S/labels
  $PY - <<'PY' 2>&1 | tail -3 | tee -a $LOG
from ultralytics import RTDETR
from pathlib import Path
m = RTDETR("weights_val/rtdetr_seed0.pt")
out = Path("eval-val/rtdetr_tdlav4/labels")
res = m.predict(source="dataset_tdlav4_tam2col/images/val", conf=0.05,
                imgsz=1024, stream=True, verbose=False)
n = 0
for r in res:
    stem = Path(r.path).stem
    lines = []
    for b in r.boxes:
        c = int(b.cls[0]); cf = float(b.conf[0])
        x, y, w, h = b.xywhn[0].tolist()
        lines.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {cf:.5f}")
    (out / f"{stem}.txt").write_text("\n".join(lines)); n += 1
print("rtdetr wrote", n)
PY
  touch eval-val/$S/labels/.done
fi
aws s3 sync eval-val/$S $VS3/$S --quiet
echo "RTDETR_VAL_DONE $(ls eval-val/$S/labels/*.txt 2>/dev/null | wc -l)" | tee -a $LOG

# 2) RF-DETR-Large (rfdetr venv, conf floor 0.01, shape 1024)
S=rfdetr_tdlav4
if [ ! -f eval-val/$S/labels/.done ]; then
  aws s3 cp $S3/weights/rfdetr_tdlav4_tam2col_best_ema.pth weights_val/rfdetr_best_ema.pth --quiet
  $RF code/rfdetr_predict.py --checkpoint weights_val/rfdetr_best_ema.pth \
      --source $IMG --out eval-val/$S --conf 0.01 --shape 1024 2>&1 | tail -3 | tee -a $LOG
  touch eval-val/$S/labels/.done
fi
aws s3 sync eval-val/$S $VS3/$S --quiet
echo "RFDETR_VAL_DONE $(ls eval-val/$S/labels/*.txt 2>/dev/null | wc -l)" | tee -a $LOG

# 3) Docling heron fine-tune (conf floor 0.05)
S=docling_heron_tdlav4
if [ ! -f eval-val/$S/labels/.done ]; then
  aws s3 sync $S3/weights/docling_heron_tdlav4_tam2col weights_val/heron_ft --quiet
  $PY code/docling_heron_predict.py --source $IMG --out eval-val/$S \
      --model weights_val/heron_ft --conf 0.05 --batch 4 2>&1 | tail -3 | tee -a $LOG
  touch eval-val/$S/labels/.done
fi
aws s3 sync eval-val/$S $VS3/$S --quiet
echo "HERON_FT_VAL_DONE $(ls eval-val/$S/labels/*.txt 2>/dev/null | wc -l)" | tee -a $LOG

# 4) Docling heron off-the-shelf (conf floor 0.05)
S=docling_heron_ots
if [ ! -f eval-val/$S/labels/.done ]; then
  $PY code/docling_heron_predict.py --source $IMG --out eval-val/$S \
      --model docling-project/docling-layout-heron --conf 0.05 --batch 4 2>&1 | tail -3 | tee -a $LOG
  touch eval-val/$S/labels/.done
fi
aws s3 sync eval-val/$S $VS3/$S --quiet
echo "HERON_OTS_VAL_DONE $(ls eval-val/$S/labels/*.txt 2>/dev/null | wc -l)" | tee -a $LOG

echo "=== $(date -u) BOXA ALL DONE ===" | tee -a $LOG
touch /home/ubuntu/tdlav4/valA.complete
