#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-.}"
NUM_IMAGES="${2:-1000}"
BASE_DIR="${3:-"$ROOT_DIR/data/refcoco"}"
ANN_ID="${4:-jxu124/refcoco}"

COCO_ID="${5:-visual-layer/coco-2014-vl-enriched}"

for seed in {1..10}; do
  subset="train_${NUM_IMAGES}_seed${seed}"
  echo ">>> Seed ${seed} → subset ${subset}"
  python download_dataset.py \
    --mode train \
    --root "${ROOT_DIR}" \
    --base-dir "${BASE_DIR}" \
    --num-images "${NUM_IMAGES}" \
    --seed "${seed}" \
    --subset-name "${subset}" \
    --ann-dataset-id "${ANN_ID}" \
    --coco-dataset-id "${COCO_ID}" \
    --overwrite
done
