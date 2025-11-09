
# https://github.com/seilk/LocalizationHeads
# 依赖（Transformers 要新一点；禁用闪存注意力以便拿到注意力张量）
pip install -U
pip install "transformers>=4.52.3" accelerate datasets pycocotools hydra-core hydra-colorlog pillow tqdm matplotlib
pip install -r requirements.txt

# 1) GPU & 挂载 Drive（可选但强烈建议）
nvidia-smi

