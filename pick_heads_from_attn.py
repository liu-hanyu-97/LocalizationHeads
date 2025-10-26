import os, glob, math, pickle, json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from scipy.ndimage import label as cc_label  # 连通域

def load_attn(pkl_path):
    """兼容 np.load / pickle.load 保存的两种格式。
    期望返回 (attn[L,H,1,V], meta_dict或None)
    """
    obj = np.load(pkl_path, allow_pickle=True)
    attn = obj['attn']
    # 转换为 np.ndarray
    attn = np.array(attn)
    assert attn.ndim == 4, f"{pkl_path} attn.ndim={attn.ndim}, 期望 [L,H,1,V]"
    return attn

def elbow_threshold(sorted_vals):
    """简单肘点：取升序数组与两端连线的最大垂距点"""
    x = np.arange(len(sorted_vals))
    y = sorted_vals
    # 直线 y = y0 + (yN - y0) * (x-x0)/(xN-x0)
    y0, yN = y[0], y[-1]
    y_line = y0 + (yN - y0) * (x - x[0]) / (x[-1] - x[0] + 1e-9)
    dist = (y - y_line)
    idx = np.argmax(dist)
    return y[idx]

def spatial_entropy(a_1d):
    """a_1d: 长度 V 的注意力，对应图像 token。
    1) reshape -> P×P；2) 按均值二值化；3) 8邻域连通域；4) H=-∑p log p
    """
    V = a_1d.shape[0]
    P = int(round(math.sqrt(V)))
    if P*P != V or P <= 1:
        # 不合法尺寸，返回高熵让它不会被选
        return 1e9
    A = a_1d.reshape(P, P)
    thr = float(A.mean())
    B = (A > thr).astype(np.uint8)
    if B.sum() == 0:
        return 1e9
    # 8 邻域
    structure = np.ones((3,3), dtype=np.uint8)
    lbl, ncomp = cc_label(B, structure=structure)
    if ncomp == 0:
        return 1e9
    sizes = np.array([(lbl==i).sum() for i in range(1, ncomp+1)], dtype=float)
    probs = sizes / sizes.sum()
    H = -(probs * np.log(probs + 1e-12)).sum()
    return float(H)

def pick_heads_from_pkls(
    root_dir,
    topk=3,
    per_sample=10,
    query_index=0,
):
    root = Path(root_dir)
    # 不包含里面带有meta的文件
    pkls = sorted(glob.glob(str(root/"**/*.pkl"), recursive=True))
    pkls = [p for p in pkls if "meta" not in p]
    print(f"Found {len(pkls)} .pkl files under {root}")

    assert pkls, f"找不到任何 .pkl：{root}"

    # ---------- Pass 1: 平均 Attention Sum & 肘点筛选 ----------
    S_sums = None  # shape [L,H]
    n_files = 0
    V_first = None
    for p in pkls:
        attn = load_attn(p)
        L, H, Q, V = attn.shape
        if V_first is None:
            V_first = V
        # 可选：如果你的预处理保证 V 恒定，可以跳过这个判断
        if V != V_first:
            # 图像token数不一致就跳过该样本（或也可以先插值到统一尺寸）
            continue
        assert Q > query_index, f"{p} 的 query 维度={Q}，取不到 index={query_index}"
        # 只取图片 tokens 的权重总和：sum_{V}
        S_img = attn[:,:,query_index,:].sum(axis=-1)  # [L,H]
        if S_sums is None:
            S_sums = np.zeros_like(S_img, dtype=np.float64)
        S_sums += S_img
        n_files += 1
    assert n_files > 0, "没有有效样本用于统计平均 Attention Sum"
    S_bar = S_sums / n_files  # [L,H]
    flat = np.sort(S_bar.flatten())
    tau = elbow_threshold(flat)

    keep_mask = (S_bar >= tau)  # [L,H]
    kept_heads = [(l,h) for l in range(S_bar.shape[0]) for h in range(S_bar.shape[1]) if keep_mask[l,h]]
    print(f"[Pass1] 样本数={n_files}, tau={tau:.6f}, 保留头数={len(kept_heads)}/{S_bar.size}")

    # ---------- Pass 2: 空间熵 + 选择频率 ----------
    counter = Counter()
    skipped = 0
    for p in pkls:
        attn = load_attn(p)
        L, H, Q, V = attn.shape
        if V != V_first:
            skipped += 1
            continue
        ent_list = []
        for (l,h) in kept_heads:
            a = attn[l, h, query_index, :]  # [V]
            H_sp = spatial_entropy(a)
            ent_list.append((H_sp, l, h))
        # 取熵最小的前 per_sample 个头
        ent_list.sort(key=lambda x: x[0])
        for _, l, h in ent_list[:per_sample]:
            counter[(l,h)] += 1
    if skipped:
        print(f"[Pass2] 跳过了 {skipped} 个样本（V 不一致）")

    total_votes = sum(counter.values())
    ranked = counter.most_common()
    top_fixed = [{"layer": int(l), "head": int(h)} for (l,h), _ in ranked[:topk]]
    # 也给出频率表（前100）
    freq_table = [
        {"layer": int(l), "head": int(h), "count": int(c), "freq": float(c/total_votes)}
        for (l,h), c in ranked[:100]
    ]
    out = {
        "summary": {
            "num_samples": n_files,
            "kept_heads": len(kept_heads),
            "tau": float(tau),
            "topk": topk,
            "per_sample": per_sample,
        },
        "top_fixed": top_fixed,
        "freq_table": freq_table,
    }
    return out

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="包含训练样本 .pkl 的目录（递归搜索）")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--per_sample", type=int, default=10)
    ap.add_argument("--query_index", type=int, default=0)
    ap.add_argument("--save", type=str, default="")
    args = ap.parse_args()

    res = pick_heads_from_pkls(
        args.root,
        topk=args.topk,
        per_sample=args.per_sample,
        query_index=args.query_index,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print("Saved:", args.save)
