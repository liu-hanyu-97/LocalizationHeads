#!/usr/bin/env python3
"""
COCO test2014 + RefCOCO / RefCOCO+ / RefCOCOg 下载器 & 训练子集构建（精简版）
- 仅依赖: datasets, pillow, tqdm
- 已存在文件自动跳过
- 支持：download / subset 两个子命令（不写子命令默认等同于 download）

用法示例：
  # 与原脚本兼容的常见用法
  python coco_refcoco_cli_slim.py --datasets refcoco --root /content
  python coco_refcoco_cli_slim.py download -d coco refcoco refcocoplus refcocog --root /content
  python coco_refcoco_cli_slim.py download -d refcocog --splits val test
  python coco_refcoco_cli_slim.py subset --num-images 1000 --base-dir /content/refcoco \
      --ann-dataset-id jxu124/refcoco --coco-dataset-id visual-layer/coco-2014-vl-enriched \
      --subset-name train_1000 --one-bbox-policy random

可选加速：
  pip install -U hf_transfer  # 然后使用 --hf-transfer（默认开启）
"""
import os
import re
import sys
import json
import random
import argparse
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional

from PIL import Image
from tqdm.auto import tqdm
from datasets import load_dataset, get_dataset_split_names

# -------------------------------------------------
# 常量
# -------------------------------------------------
REF_DATASETS = {
    "refcoco": "lmms-lab/RefCOCO",
    "refcocoplus": "lmms-lab/RefCOCOplus",
    "refcocog": "lmms-lab/RefCOCOg",
}
DEFAULT_REF_SPLITS_ORDER = ["val", "testA", "testB", "test"]

# -------------------------------------------------
# 工具
# -------------------------------------------------

def _json_safe(v):
    """尽量把 HF 的类型转成 JSON 可序列化。"""
    try:
        json.dumps(v)
        return v
    except TypeError:
        if isinstance(v, (set,)):
            return list(v)
        if isinstance(v, (bytes, bytearray)):
            return v.decode("utf-8", errors="ignore")
        return str(v)


def save_annotations_jsonl(ds, out_path: Path, drop_image_col: str = "image", overwrite: bool = False) -> None:
    """把数据集去掉 image 列后写成 JSONL（逐行），避免大文件。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        print(f"✅ 标注已存在：{out_path}")
        return

    cols = [c for c in ds.column_names if c != drop_image_col]
    print(f"✍️  保存标注 → {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in ds.select_columns(cols):
            f.write(json.dumps({k: _json_safe(v) for k, v in ex.items()}, ensure_ascii=False) + "\n")
    print(f"✅ 标注已保存：{out_path}")


def save_annotations_jsonl_iter(ds_iter: Iterable[dict], out_path: Path, drop_image_col: str = "image", overwrite: bool = False, limit: Optional[int] = None) -> None:
    """Write annotations from an iterable dataset (e.g., streaming) to JSONL, dropping the image column."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        print(f"ℹ️  注释已存在：{out_path}")
        return

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, ex in enumerate(ds_iter):
            if limit is not None and idx >= limit:
                break
            rec = {k: _json_safe(v) for k, v in ex.items() if k != drop_image_col}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    print(f"? 注释已保存：{out_path}（{written} 行）")


def _default_name(ex: Dict, idx: int, prefix: str = "img_") -> str:
    name = ex.get("file_name")
    if not name:
        name = f"{prefix}{idx:06d}.jpg"
    if "." not in Path(name).name:
        name = f"{name}.jpg"
    return Path(name).name  # 防止路径穿越


def save_images(
    ds_iter: Iterable[dict],
    out_dir: Path,
    *,
    name_fn: Optional[Callable[[dict, int], str]] = None,
    image_col: str = "image",
    overwrite: bool = False,
    total: Optional[int] = None,
    limit: Optional[int] = None,
) -> None:
    """把可迭代的样本（含 PIL 图像）保存到磁盘。支持 streaming 和非 streaming。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 尝试自动计算总数（非 streaming）
    if total is None:
        try:
            total = len(ds_iter)  # type: ignore[arg-type]
        except Exception:
            pass

    written = skipped = 0
    with tqdm(total=(limit if limit is not None else total), desc=f"保存图片 → {out_dir.name}") as bar:
        for idx, ex in enumerate(ds_iter):
            if limit is not None and idx >= limit:
                break
            img: Image.Image = ex[image_col]
            fname = (name_fn or _default_name)(ex, idx)
            out_path = out_dir / fname
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if out_path.exists() and not overwrite:
                skipped += 1
                bar.update(1)
                continue
            try:
                img.save(out_path, format="JPEG")
                written += 1
            except Exception as e:
                print(f"⚠️  保存失败 {out_path}: {e}")
            bar.update(1)

    print(f"✅ 图片完成：写入 {written}，跳过 {skipped}，目标目录 {out_dir}")


# -------------------------------------------------
# 下载：COCO test2014
# -------------------------------------------------

def download_coco_test(out_images_dir: Path, out_ann_dir: Path, *, overwrite: bool = False, limit: Optional[int] = None) -> None:
    print("==== visual-layer/coco-2014-vl-enriched :: test ====")
    ds = load_dataset("visual-layer/coco-2014-vl-enriched", split="test")

    def coco_test_name(ex: dict, _i: int) -> str:
        iid = ex.get("image_id")
        if iid is None:
            return f"COCO_test2014_{_i:06d}.jpg"
        return f"COCO_test2014_{int(iid):012d}.jpg"  # 12 位 zero-pad，更标准

    save_images(ds, out_images_dir, name_fn=coco_test_name, image_col="image", overwrite=overwrite, total=len(ds), limit=limit)
    save_annotations_jsonl(ds, out_ann_dir / "test.jsonl", drop_image_col="image", overwrite=overwrite)


# -------------------------------------------------
# 下载：RefCOCO / RefCOCO+ / RefCOCOg
# -------------------------------------------------

def download_ref_like(dataset_key: str, base_dir: Path, *, splits: Optional[List[str]] = None, overwrite: bool = False, limit: Optional[int] = None) -> None:
    dataset_id = REF_DATASETS[dataset_key]
    available = get_dataset_split_names(dataset_id)
    target_splits = [sp for sp in (splits or DEFAULT_REF_SPLITS_ORDER) if sp in available]

    if not target_splits:
        print(f"⚠️  未发现可用切分（{dataset_id}）")
        return

    for sp in target_splits:
        print(f"==== {dataset_id} :: {sp} ====")
        ds_stream_for_images = load_dataset(dataset_id, split=sp, streaming=True)
        out_images = base_dir / sp
        save_images(
            ds_stream_for_images,
            out_images,
            name_fn=lambda ex, i: _default_name(ex, i),
            image_col="image",
            overwrite=overwrite,
            limit=limit,
        )
        ds_stream_for_ann = load_dataset(dataset_id, split=sp, streaming=True)
        save_annotations_jsonl_iter(ds_stream_for_ann, base_dir / f"{sp}.jsonl", drop_image_col="image", overwrite=overwrite, limit=limit)


# -------------------------------------------------
# 构建 RefCOCO 训练子集（流式下载 COCO train2014）
# -------------------------------------------------

def _pad12(x) -> str:
    s = re.sub(r"\D", "", str(x))
    return f"{int(s):012d}"


def _id12_from_fname(fn: str) -> Optional[str]:
    m = re.search(r"([0-9]{12})", str(fn))
    return m.group(1) if m else None


def build_refcoco_train_subset(
    *,
    num_images: int = 1000,
    seed: int = 42,
    ann_dataset_id: str = "jxu124/refcoco",
    coco_dataset_id: str = "visual-layer/coco-2014-vl-enriched",
    base_dir: Path = Path("refcoco"),
    subset_name: str = "train_1000",
    one_bbox_policy: str = "random",  # random | largest | first
    overwrite: bool = False,
) -> None:
    print(f"==== 构建 RefCOCO 训练子集：{num_images} 张唯一图片，每图 1 条标注 ====")

    # 1) 训练标注（不含图像）
    ann = load_dataset(ann_dataset_id, split="train")

    # image_id → indices
    if "image_id" in ann.column_names:
        ids12 = [_pad12(x) for x in ann["image_id"]]
    elif "file_name" in ann.column_names:
        ids12 = [_id12_from_fname(fn) for fn in ann["file_name"]]
    else:
        raise ValueError(f"{ann_dataset_id} 缺少 image_id/file_name")

    by_id: Dict[str, List[int]] = {}
    for i, iid in enumerate(ids12):
        if iid is None:
            continue
        by_id.setdefault(iid, []).append(i)

    uniq_ids = list(by_id.keys())
    if num_images > len(uniq_ids):
        print(f"⚠️  目标 {num_images} > 可用唯一图片 {len(uniq_ids)}，自动调小。")
        num_images = len(uniq_ids)

    rng = random.Random(seed)
    chosen_ids = rng.sample(uniq_ids, k=num_images)

    def pick_one(idxs: List[int]) -> int:
        if one_bbox_policy == "largest" and "bbox" in ann.column_names:
            return max(idxs, key=lambda j: (ann[j]["bbox"][2] * ann[j]["bbox"][3]) if ann[j]["bbox"] else -1.0)
        if one_bbox_policy == "first":
            return idxs[0]
        return rng.choice(idxs)

    chosen_indices = [pick_one(by_id[iid]) for iid in chosen_ids]
    ann_subset = ann.select(chosen_indices)

    # 2) 保存子集标注
    out_images_dir = base_dir / subset_name
    save_annotations_jsonl(ann_subset, base_dir / f"{subset_name}.jsonl", drop_image_col="image", overwrite=overwrite)

    # 3) 仅下载命中的 COCO train2014 图片（streaming）
    coco_stream = load_dataset(coco_dataset_id, split="train", streaming=True)
    chosen_set = set(chosen_ids)

    def iter_matching() -> Iterator[dict]:
        remaining = set(chosen_set)
        for ex in coco_stream:
            iid12 = f"{int(ex['image_id']):012d}"
            if iid12 in remaining:
                if not ex.get("file_name"):
                    ex["file_name"] = f"COCO_train2014_{iid12}.jpg"
                remaining.remove(iid12)
                yield ex
            if not remaining:
                break

    save_images(
        iter_matching(),
        out_images_dir,
        name_fn=lambda ex, i: Path(ex.get("file_name") or f"COCO_train2014_{int(ex['image_id']):012d}.jpg").name,
        image_col="image",
        overwrite=overwrite,
        total=len(chosen_ids),
    )

    print(f"✅ 子集完成：{len(chosen_ids)} 张唯一图片，{len(ann_subset)} 条标注（每图 1 条）")
    print(f"   - 图片 → {out_images_dir}")
    print(f"   - 标注 → {base_dir / (subset_name + '.jsonl')}")


# -------------------------------------------------
# CLI
# -------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="COCO test2014 + RefCOCO/+/g 下载 & 训练子集构建（精简 CLI）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root", type=Path, default=Path("."), help="输出根目录")
    p.add_argument("--hf-transfer", dest="hf_transfer", action="store_true", help="开启 hf_transfer 传输加速（若已安装）")
    p.add_argument("--no-hf-transfer", dest="hf_transfer", action="store_false", help="关闭 hf_transfer 传输加速")
    p.set_defaults(hf_transfer=True)

    # download test
    p.add_argument("--mode", choices=["test", "train"], default="test", help="下载模式：test（下载 COCO test2014 + Ref*）或 train（构建 RefCOCO 训练子集）")
    p.add_argument("--datasets", "-d", nargs="+", choices=["coco", "refcoco", "refcocoplus", "refcocog"], default=["refcoco"], help="选择要下载的数据集")
    p.add_argument("--coco-dir", type=Path, default=None, help="COCO test2014 图片输出目录（默认 root/data/coco_images/test2014）")
    p.add_argument("--refcoco-dir", type=Path, default=None, help="RefCOCO 输出目录（默认 root/data/refcoco）")
    p.add_argument("--refcocoplus-dir", type=Path, default=None, help="RefCOCO+ 输出目录（默认 root/data/refcocoplus）")
    p.add_argument("--refcocog-dir", type=Path, default=None, help="RefCOCOg 输出目录（默认 root/data/refcocog）")
    p.add_argument("--splits", nargs="+", choices=["val", "testA", "testB", "test"], help="Ref* 数据集的目标切分")
    p.add_argument("--overwrite", action="store_true", help="覆盖同名文件/标注")
    p.add_argument("--limit", type=int, default=None, help="仅下载前 N 张图片（调试用）")

    # train subset
    p.add_argument("--num-images", type=int, default=1000, help="唯一图片数量（每图 1 条标注）")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--ann-dataset-id", type=str, default="jxu124/refcoco", help="训练标注数据集 ID（不含图像）")
    p.add_argument("--coco-dataset-id", type=str, default="visual-layer/coco-2014-vl-enriched", help="COCO 2014 数据集 ID（含图像）")
    p.add_argument("--base-dir", type=Path, default=None, help="输出基目录（默认 root/refcoco）")
    p.add_argument("--subset-name", type=str, default="train_1000", help="子集名称（图片目录名 & 标注文件名前缀）")
    p.add_argument("--one-bbox-policy", choices=["random", "largest", "first"], default="random", help="每图选择哪一条标注")

    return p


def resolve_dirs(args):
    root: Path = args.root
    coco_dir = getattr(args, "coco_dir", None) or (root / "data" / "coco_images" / "test2014")
    refcoco_dir = getattr(args, "refcoco_dir", None) or (root / "data" / "refcoco")
    refcocoplus_dir = getattr(args, "refcocoplus_dir", None) or (root / "data" / "refcocoplus")
    refcocog_dir = getattr(args, "refcocog_dir", None) or (root / "data" / "refcocog")
    for p in [coco_dir, refcoco_dir, refcocoplus_dir, refcocog_dir]:
        p.mkdir(parents=True, exist_ok=True)
    return coco_dir, refcoco_dir, refcocoplus_dir, refcocog_dir


def main() -> None:

    parser = build_parser()
    args = parser.parse_args()

    # hf_transfer 环境变量（未安装不会报错）
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" if args.hf_transfer else "0"

    mode = args.mode

    if mode == "test":
        coco_dir, refcoco_dir, refcocoplus_dir, refcocog_dir = resolve_dirs(args)

        if "coco" in args.datasets:
            download_coco_test(
                out_images_dir=coco_dir,
                out_ann_dir=coco_dir,
                overwrite=args.overwrite,
                limit=args.limit,
            )

        if "refcoco" in args.datasets:
            download_ref_like("refcoco", base_dir=refcoco_dir, splits=args.splits, overwrite=args.overwrite, limit=args.limit)
        if "refcocoplus" in args.datasets:
            download_ref_like("refcocoplus", base_dir=refcocoplus_dir, splits=args.splits, overwrite=args.overwrite, limit=args.limit)
        if "refcocog" in args.datasets:
            download_ref_like("refcocog", base_dir=refcocog_dir, splits=args.splits, overwrite=args.overwrite, limit=args.limit)

        print("🎉 全部完成：")
        print(f"- COCO test2014 图片 → {coco_dir}")
        print(f"- RefCOCO 图片/标注 → {refcoco_dir}")
        print(f"- RefCOCO+ 图片/标注 → {refcocoplus_dir}")
        print(f"- RefCOCOg 图片/标注 → {refcocog_dir}")

    elif mode == "train":
        # 基目录默认与下载时一致（root/refcoco）
        _, refcoco_dir, _, _ = resolve_dirs(argparse.Namespace(root=args.root, coco_dir=None, refcoco_dir=None, refcocoplus_dir=None, refcocog_dir=None))
        base_dir = args.base_dir or refcoco_dir
        base_dir.mkdir(parents=True, exist_ok=True)

        build_refcoco_train_subset(
            num_images=args.num_images,
            seed=args.seed,
            ann_dataset_id=args.ann_dataset_id,
            coco_dataset_id=args.coco_dataset_id,
            base_dir=base_dir,
            subset_name=args.subset_name,
            one_bbox_policy=args.one_bbox_policy,
            overwrite=args.overwrite,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
