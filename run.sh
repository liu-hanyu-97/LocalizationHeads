# test
set -e
MODEL_NAME="deepseek-ai/deepseek-vl-1.3b-chat"
CONV_MODE="chatml_direct"     # DeepSeek-VL 使用 ChatML 模板


python prepare_jsonl.py \
  --input_jsonl "data/refcoco/testA.jsonl" \
  --image_root "data/refcoco/testA/"  \
  --output_jsonl "data/refcoco/testA_prompts.jsonl" \
  --prompt_mode answer_first --answer_index 0 --id_start 0

tail data/refcoco/testA_prompts.jsonl

# Train
python prepare_jsonl.py \
  --is_train \
  --input_jsonl "data/refcoco/train_1000.jsonl" \
  --image_root "data/refcoco/train_1000/"  \
  --output_jsonl "data/refcoco/train_1000_prompts.jsonl" \
  --prompt_mode answer_all --answer_index 0 --id_start 0

# for seed in 1-10 run 
for seed in {1..10}; do
  subset="train_1000_seed${seed}"
  python prepare_jsonl.py \
    --is_train \
    --input_jsonl "data/refcoco/${subset}.jsonl" \
    --image_root "data/refcoco/${subset}/"  \
    --output_jsonl "data/refcoco/${subset}_prompts.jsonl" \
    --prompt_mode answer_all --answer_index 0 --id_start 0
done


# get attention heads for ten seeds
for seed in {1..10}; do
  subset="train_1000_seed${seed}"
  echo ">>> Seed ${seed} → subset ${subset}"
  python pipeline.py \
    stage=batch \
    data.data_file="data/refcoco/${subset}_prompts.jsonl" \
    data.process_all=true \
    data.output_dir="outputs/${subset}/" \
    model.name="${MODEL_NAME}" \
    model.conv_mode=${CONV_MODE} \
    model.use_generate=False
done
