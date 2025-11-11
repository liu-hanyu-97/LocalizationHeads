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


# ===============================
# 将 RefCOCO 训练 JSONL 转为“测试集风格”
# 输入:  /content/refcoco/train_1000.jsonl
# 输出:  /content/refcoco/train_1000_testfmt.jsonl
# ===============================
import json, math
from pathlib import Path

def _to_float_list(xs):
    out = []
    for v in xs:
        try:
            out.append(float(v))
        except Exception:
            # 若有嵌套字符串等，直接跳过
            continue
    return out

def _first_polygon(seg):
    """
    COCO segmentation 可能是: [ [x1,y1,...], [x1,y1,...], ... ] 或直接 [x1,y1,...]
    统一取第一段 polygon，并展开为一维列表
    """
    if isinstance(seg, list) and len(seg) > 0:
        if isinstance(seg[0], list):  # 多段
            return _to_float_list(seg[0])
        else:  # 已是一维
            return _to_float_list(seg)
    return []

def convert_refcoco_train_to_testfmt(in_jsonl: Path, out_jsonl: Path):
    QUESTION_TEXT = "Please carefully observe the area circled in the image and come up with a caption for the area."
    cnt_in, cnt_out = 0, 0
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with open(in_jsonl, "r", encoding="utf-8") as fin, \
         open(out_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            cnt_in += 1
            ex = json.loads(line)

            # 1) 解析 raw_anns（字符串形式）
            raw_anns = ex.get("raw_image_info", None)
            if isinstance(raw_anns, str):
                try:
                    raw_anns = json.loads(raw_anns)
                except Exception:
                    raw_anns = {}
            elif not isinstance(raw_anns, dict):
                raw_anns = {}

            # 2) segmentation
            seg = raw_anns.get("segmentation", ex.get("segmentation"))
            seg = _first_polygon(seg) or []

            # 3) bbox（优先 raw_anns['bbox'] 的 [x,y,w,h]）
            bbox = raw_anns.get("bbox")
            if bbox is None:
                bbox = ex.get("bbox")


            # 4) iscrowd
            iscrowd = int(raw_anns.get("iscrowd", 0))

            # 5) file_name
            file_name = raw_anns.get("file_name")
            if isinstance(file_name, str) and "/" in file_name:
                # 保留与 test.jsonl 相同的短文件名风格
                file_name = Path(file_name).name

            # 6) answers（用 captions 或 sentences[*].sent）
            answers = ex.get("captions")
            if not answers:
                sentences = ex.get("sentences", [])
                if isinstance(sentences, list):
                    answers = [s.get("sent") or s.get("raw") for s in sentences if isinstance(s, dict)]
                    answers = [a for a in answers if a]
                else:
                    answers = []
            # 兜底：没有文本就用空串，避免空列表
            if not answers:
                answers = [""]

            # 7) question_id：用 ann_id（如无则用 ref_id / image_id / 行号）
            qid = ex.get("ann_id") or ex.get("ref_id") or ex.get("image_id") or cnt_in
            qid = str(qid)

            out = {
                "question_id": qid,
                "question": QUESTION_TEXT,
                "answer": answers,
                "segmentation": seg,   # 一维坐标列表
                "bbox": bbox,          # [x, y, w, h]
                "iscrowd": iscrowd,
                "file_name": file_name
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            cnt_out += 1

    print(f"✅ 转换完成：输入 {cnt_in} 行 → 输出 {cnt_out} 行")
    print(f"   - 输出文件：{out_jsonl}")

def main():
    parser = argparse.ArgumentParser(description="Convert RefCOCO-style JSONL to prompt+image JSONL")
    parser.add_argument("--input_jsonl", required=True, help="Path to input JSONL file")
    parser.add_argument("--image_root", required=True, help="Directory containing images referenced by file_name")
    parser.add_argument("--output_jsonl", required=True, help="Path to write output JSONL")
    parser.add_argument("--prompt_mode", choices=["answer_first", "answer_all"], default="answer_first",
                        help="How to form the prompt from answers")
    parser.add_argument("--answer_index", type=int, default=0, help="Index of answer to use when answer_first")
    parser.add_argument("--id_start", type=int, default=0, help="Starting ID for generated items")

    parser.add_argument("--is_train", action="store_true",
                        default=False, help="Whether to convert train JSONL to test format")

    args = parser.parse_args()

    if args.is_train:
        in_path = Path(args.input_jsonl)
        old_in_path = in_path
        in_path = Path(args.input_jsonl.replace(".jsonl", "_testfmt.jsonl"))
        convert_refcoco_train_to_testfmt(old_in_path, in_path)
    else:
        in_path = Path(args.input_jsonl)
    convert_refcoco_to_prompt_image(
        input_jsonl=str(in_path),
        image_root=args.image_root,
        output_jsonl=args.output_jsonl,
        prompt_mode=args.prompt_mode,
        answer_index=args.answer_index,
        id_start=args.id_start,
    )
    if args.is_train:
        import os
        os.remove(str(in_path))


if __name__ == "__main__":
    main()
    