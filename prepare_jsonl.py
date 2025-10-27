import json
import argparse
from pathlib import Path

def convert_refcoco_to_prompt_image(
    input_jsonl: str,
    image_root: str,
    output_jsonl: str,
    *,
    prompt_mode: str = "answer_first", # prompt 生成模式，可选 "answer_select" 或 "answer_all"
    answer_index: int = 0,          # 用第几个答案（默认第 1 个）
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

    n_in, n_out = 0, 0
    next_id = id_start

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        lines = fin.readlines()
        print(f"总行数: {len(lines)}")
        for line in lines:
            line = line.strip()
            if not line:
                print("空行，跳过")
                continue
            n_in += 1
            try:
                rec = json.loads(line)
            except Exception as e:
                print(f"❌ 解析失败: {e}")
                continue

            # 选 prompt
            prompt = ""
            if prompt_mode == "answer_first" and isinstance(rec.get("answer"), list):
                # 去掉空白并去重，取第 answer_index 个
                answers = [str(a).strip() for a in rec["answer"] if isinstance(a, str) and str(a).strip()]
                answers = list(dict.fromkeys(answers))  # 保留顺序去重
                if answers and 0 <= answer_index < len(answers):
                    prompt = answers[answer_index]
            elif prompt_mode == "answer_all" and isinstance(rec.get("answer"), list):
                answers = [str(a).strip() for a in rec["answer"] if isinstance(a, str) and str(a).strip()]
                answers = list(dict.fromkeys(answers))  # 保留顺序去重
                if answers:
                    prompt = ", ".join(answers)
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
                "question_id": rec.get("question_id"),
                "prompt": prompt,
                "image": str(img_path),
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            next_id += 1
            n_out += 1

    print(f"✅ 转换完成：读入 {n_in} 行 → 写出 {n_out} 行 → {out_path}")
    return

def main():
    parser = argparse.ArgumentParser(description="Convert RefCOCO-style JSONL to prompt+image JSONL")
    parser.add_argument("--input_jsonl", required=True, help="Path to input JSONL file")
    parser.add_argument("--image_root", required=True, help="Directory containing images referenced by file_name")
    parser.add_argument("--output_jsonl", required=True, help="Path to write output JSONL")
    parser.add_argument("--prompt_mode", choices=["answer_first", "answer_all"], default="answer_first",
                        help="How to form the prompt from answers")
    parser.add_argument("--answer_index", type=int, default=0, help="Index of answer to use when answer_first")
    parser.add_argument("--id_start", type=int, default=0, help="Starting ID for generated items")

    args = parser.parse_args()
    convert_refcoco_to_prompt_image(
        input_jsonl=args.input_jsonl,
        image_root=args.image_root,
        output_jsonl=args.output_jsonl,
        prompt_mode=args.prompt_mode,
        answer_index=args.answer_index,
        id_start=args.id_start,
    )


if __name__ == "__main__":
    main()

# Example usage:
# convert_refcoco_to_prompt_image(
#     input_jsonl="/content/refcoco/testA.jsonl",
#     image_root="/content/refcoco/testA",
#     output_jsonl="/content/refcoco/testA_prompts.jsonl",
#     prompt_mode="answer_first", # 优先用 answer 作为 prompt
#     answer_index=0,              # 取第一个答案
#     id_start=0
# )
