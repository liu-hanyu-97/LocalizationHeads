import os, json
from pathlib import Path

def convert_refcoco_to_prompt_image(
    input_jsonl: str,
    image_root: str,
    output_jsonl: str,
    *,
    image_key: str = "image",       # 你示例里叫 "image"；如果要配合某些代码可改成 "image_path"
    use_answer_first: bool = True,  # 优先用 answer 列表；设为 False 则直接用 question
    answer_index: int = 0,          # 用第几个答案（默认第 1 个）
    relative_to: str | None = None, # 如果想把 image 写成相对路径，可传入根目录，比如 "/content"
    id_start: int = 0               # 起始 id
):
    in_path = Path(input_jsonl)
    img_root = Path(image_root)
    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def ensure_punct(s: str) -> str:
        s = (s or "").strip()
        if not s:
            return s
        if s[-1] not in ".!?。！？…~":
            s += "."
        return s

    def to_rel(path: Path) -> str:
        if relative_to:
            try:
                return str(path.resolve().relative_to(Path(relative_to).resolve()))
            except Exception:
                return str(path)
        return str(path)

    n_in, n_out = 0, 0
    next_id = id_start

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue

            # 选 prompt
            prompt = ""
            if use_answer_first and isinstance(rec.get("answer"), list):
                # 去掉空白并去重，取第 answer_index 个
                answers = [str(a).strip() for a in rec["answer"] if isinstance(a, str) and str(a).strip()]
                answers = list(dict.fromkeys(answers))  # 保留顺序去重
                if answers and 0 <= answer_index < len(answers):
                    prompt = answers[answer_index]
            if not prompt:
                prompt = str(rec.get("question", "")).strip()
            prompt = ensure_punct(prompt)
            if not prompt:
                continue  # 没有可用文本就跳过

            # 拼 image 路径（使用 file_name 的 basename）
            fname = Path(str(rec.get("file_name", ""))).name
            if not fname:
                continue
            img_path = img_root / fname
            if not img_path.exists():
                # 找不到图就跳过该样本
                continue

            obj = {
                "id": next_id,
                "prompt": prompt,
                image_key: to_rel(img_path),
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            next_id += 1
            n_out += 1

    print(f"✅ 转换完成：读入 {n_in} 行 → 写出 {n_out} 行 → {out_path}")
    return

# Example usage:
# convert_refcoco_to_prompt_image(
#     input_jsonl="/content/refcoco/testA.jsonl",
#     image_root="/content/refcoco/testA",
#     output_jsonl="/content/refcoco/testA_prompts.jsonl",
#     image_key="image",          # 你要的键名
#     use_answer_first=True,      # 优先用 answer 作为 prompt
#     answer_index=0,             # 取第一个答案
#     relative_to=None,     # 可选：写成相对路径，如 "refcoco/testA/COCO_....jpg"
#     id_start=0
# )

