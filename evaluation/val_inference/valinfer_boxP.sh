#!/usr/bin/env bash
# Val-split inference on the paddle box (A10G): pp ft+ots, doclayout ft+ots.
# Same predict settings / conf floor (0.05) as the v4 TEST dumps.
set -uo pipefail
export HOME=/home/ubuntu
cd /home/ubuntu/tdlav4
S3=s3://bec.bdrc.io/models/hff-detection/tdlav4
VS3=$S3/eval-val
IMG=dataset_tdlav4_tam2col/images/val
PP=/home/ubuntu/tdlav4/.venv_pp/bin/python
DL=/home/ubuntu/tdlav4/.venv_doclayout/bin/python
mkdir -p eval-val weights_val
LOG=/home/ubuntu/tdlav4/valP.log
echo "=== $(date -u) BOXP val infer start ($(ls $IMG|wc -l) val images) ===" | tee -a $LOG

# 1) PP-DocLayout-L fine-tune (paddle inference dir, conf floor 0.05)
S=pp_doclayout_tdlav4
if [ ! -f eval-val/$S/labels/.done ]; then
  $PP code2/pp_doclayout_predict.py --source $IMG --out eval-val/$S \
      --conf 0.05 --model-dir /home/ubuntu/tdlav4/pp_ft_model 2>&1 | tail -2 | tee -a $LOG
  touch eval-val/$S/labels/.done
fi
aws s3 sync eval-val/$S $VS3/$S --quiet
echo "PP_FT_VAL_DONE $(ls eval-val/$S/labels/*.txt 2>/dev/null | wc -l)" | tee -a $LOG

# 2) PP-DocLayout-L off-the-shelf (auto model, conf floor 0.05)
S=pp_doclayout_ots
if [ ! -f eval-val/$S/labels/.done ]; then
  $PP code2/pp_doclayout_predict.py --source $IMG --out eval-val/$S \
      --conf 0.05 2>&1 | tail -2 | tee -a $LOG
  touch eval-val/$S/labels/.done
fi
aws s3 sync eval-val/$S $VS3/$S --quiet
echo "PP_OTS_VAL_DONE $(ls eval-val/$S/labels/*.txt 2>/dev/null | wc -l)" | tee -a $LOG

# 3) DocLayout-YOLO fine-tune (native 4-class, conf floor 0.05)
S=doclayout_tdlav4
if [ ! -f eval-val/$S/labels/.done ]; then
  aws s3 cp $S3/weights/doclayout_tdlav4_tam2col_best.pt weights_val/doclayout_ft.pt --quiet
  $DL code2/doclayout_predict_nofuse.py --weights weights_val/doclayout_ft.pt \
      --source $IMG --out eval-val/$S --conf 0.05 --imgsz 1024 --native-classes 2>&1 | tail -2 | tee -a $LOG
  touch eval-val/$S/labels/.done
fi
aws s3 sync eval-val/$S $VS3/$S --quiet
echo "DOCLAYOUT_FT_VAL_DONE $(ls eval-val/$S/labels/*.txt 2>/dev/null | wc -l)" | tee -a $LOG

# 4) DocLayout-YOLO DocStructBench off-the-shelf (mapped classes, conf floor 0.05)
S=doclayout_docstruct_ots
BASE=weights/doclayout_yolo_docstructbench_imgsz1024.pt
if [ ! -f eval-val/$S/labels/.done ]; then
  if [ ! -f "$BASE" ]; then
    mkdir -p weights
    $DL -c "from huggingface_hub import hf_hub_download; hf_hub_download('juliozhao/DocLayout-YOLO-DocStructBench','doclayout_yolo_docstructbench_imgsz1024.pt',local_dir='weights')"
  fi
  $DL code2/doclayout_predict_nofuse.py --weights "$BASE" \
      --source $IMG --out eval-val/$S --conf 0.05 --imgsz 1024 2>&1 | tail -2 | tee -a $LOG
  touch eval-val/$S/labels/.done
fi
aws s3 sync eval-val/$S $VS3/$S --quiet
echo "DOCLAYOUT_OTS_VAL_DONE $(ls eval-val/$S/labels/*.txt 2>/dev/null | wc -l)" | tee -a $LOG

echo "=== $(date -u) BOXP ALL DONE ===" | tee -a $LOG
touch /home/ubuntu/tdlav4/valP.complete
