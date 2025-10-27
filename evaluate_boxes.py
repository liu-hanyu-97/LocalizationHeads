#!/usr/bin/env python3
"""
Simple bounding box evaluation utility.

Given a JSONL dataset containing ground-truth boxes (COCO style xywh or xyxy)
and a directory of predicted `<id>_bbox.json` files produced by this repo,
compute IoU statistics and success rates at several thresholds.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predicted bounding boxes.")
    parser.add_argument(
        "--dataset-jsonl",
        required=True,
        help="Path to the JSONL file containing ground truth annotations.",
    )
    parser.add_argument(
        "--pred-root",
        required=True,
        help="Directory that holds `<id>_bbox.json` predictions.",
    )
    parser.add_argument(
        "--iou-thresholds",
        default="0.25,0.5,0.75",
        help="Comma-separated IoU thresholds for success rates.",
    )
    return parser.parse_args()


def xywh_to_xyxy(box: Iterable[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = box
    return float(x), float(y), float(x + w), float(y + h)


def ensure_xyxy(box: Iterable[float]) -> Tuple[float, float, float, float]:
    vals = list(map(float, box))
    if len(vals) != 4:
        raise ValueError(f"Expected 4 values for bbox, got {len(vals)}")
    x0, y0, x1, y1 = vals
    if x1 < x0 or y1 < y0:
        # Assume xywh order
        return xywh_to_xyxy(vals)
    # Already xyxy but may have width/height mixed; check heuristically
    if x1 <= 1.0 and y1 <= 1.0:
        # Likely normalized xywh
        return xywh_to_xyxy(vals)
    return x0, y0, x1, y1


def iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter_w = max(0.0, ix1 - ix0)
    inter_h = max(0.0, iy1 - iy0)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def load_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse line in {path}: {exc}") from exc
    return records


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_jsonl)
    pred_root = Path(args.pred_root)
    assert dataset_path.exists(), f"Dataset JSONL not found: {dataset_path}"
    assert pred_root.exists(), f"Prediction directory not found: {pred_root}"

    thresholds = [float(t) for t in args.iou_thresholds.split(",") if t]
    thresholds.sort()

    gt_records = load_jsonl(dataset_path)
    total = len(gt_records)

    ious: List[float] = []
    missing_preds: List[str] = []
    degenerate_preds: List[str] = []

    success_counts = {thr: 0 for thr in thresholds}
    evaluated = 0
    for i, entry in enumerate(gt_records):
        sample_id = str(entry["question_id"])
        pred_path = pred_root / f"{sample_id}_bbox.json"
        if not pred_path.exists():
            missing_preds.append(sample_id)
            continue

        with pred_path.open("r", encoding="utf-8") as f:
            pred_obj = json.load(f)
            print(pred_obj)
        pred_bbox = pred_obj.get("bbox_xyxy") or pred_obj.get("bbox")
        if not pred_bbox:
            degenerate_preds.append(sample_id)
            continue

        try:
            pred_xyxy = ensure_xyxy(pred_bbox)
        except ValueError:
            degenerate_preds.append(sample_id)
            continue

        gt_bbox = entry.get("bbox_xyxy") or entry.get("bbox")
        if not gt_bbox:
            continue

        # For ground truth in COCO xywh, convert explicitly
        gt = list(map(float, gt_bbox))
        if len(gt) != 4:
            continue

        if entry.get("bbox_format", "").lower() == "xyxy":
            gt_xyxy = tuple(gt)
        else:
            gt_xyxy = xywh_to_xyxy(gt)

        val = iou(pred_xyxy, gt_xyxy)
        ious.append(val)
        evaluated += 1
        for thr in thresholds:
            if val >= thr:
                success_counts[thr] += 1

    print("=== Bounding Box Evaluation ===")
    print(f"Total samples in dataset: {total}")
    print(f"Evaluated predictions   : {evaluated}")
    print(f"Missing predictions     : {len(missing_preds)}")
    if missing_preds:
        print(f"  e.g. {missing_preds[:5]}")
    print(f"Degenerate predictions  : {len(degenerate_preds)}")
    if degenerate_preds:
        print(f"  e.g. {degenerate_preds[:5]}")

    if not ious:
        print("No valid IoU values computed. Check inputs.")
        return

    mean_iou = sum(ious) / len(ious)
    median_iou = sorted(ious)[len(ious) // 2]
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Median IoU: {median_iou:.4f}")

    for thr in thresholds:
        rate = success_counts[thr] / evaluated if evaluated else 0.0
        print(f"IoU ≥ {thr:.2f}: {success_counts[thr]} / {evaluated} = {rate:.3f}")


if __name__ == "__main__":
    main()
