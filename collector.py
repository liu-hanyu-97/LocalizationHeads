import os
import re
import pickle
from typing import Dict, Tuple

import torch
import numpy as np
from PIL import Image

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from lab.stations import MetadataStation


def _sanitize_name(s: str) -> str:
    return s.replace("/", "-").replace(" ", "_")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_image(path_or_url: str) -> Image.Image:
    if path_or_url.startswith(("http://", "https://")):
        import requests
        from io import BytesIO
        resp = requests.get(path_or_url)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    if not os.path.exists(path_or_url):
        raise FileNotFoundError(f"Image not found: {path_or_url}")
    return Image.open(path_or_url).convert("RGB")


def load_model_from_cfg(cfg) -> Tuple[object, object, object, int, str]:
    """Build model/tokenizer/image_processor using Hugging Face path.

    Returns: tokenizer, model, image_processor, context_len, model_name_str
    """
    disable_torch_init()
    # Optional user cache control (force override; must be set before downloads)
    if getattr(cfg.model, "cache_dir", None):
        cache_dir = str(cfg.model.cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        os.environ["HF_HOME"] = cache_dir
        os.environ["HF_HUB_CACHE"] = cache_dir

    # Choose device string
    device = cfg.device
    if device == "auto":
        device = f"cuda:{cfg.device_id}" if cfg.device_id >= 0 and torch.cuda.is_available() else "cpu"

    model_name_str = get_model_name_from_path(cfg.model.name)
    tok, model, img_proc, context_len = load_pretrained_model(
        model_path=cfg.model.name,
        cache_dir=cfg.model.cache_dir,
        model_base=cfg.model.base,
        model_name=model_name_str,
        device=device,
        use_flash_attn=getattr(cfg.model, "use_flash_attn", False),
    )
    return tok, model, img_proc, context_len, model_name_str


def _forward_collect(model, tokenizer, image_processor, input_ids, image_tensor, image_sizes):
    """Collect attentions via a single forward pass.

    Returns attention focused on image tokens with shape [L, H, 1, V].
    """
    outputs = model(
        input_ids=input_ids,
        images=image_tensor.unsqueeze(0),
        image_sizes=image_sizes,
        output_attentions=True,
        return_dict=True,
    )
    attn_layers = outputs.attentions  # tuple length L of [B,H,Tq,Tk]
    if not attn_layers:
        raise RuntimeError("No attentions returned from forward()")

    layers = []
    for t in attn_layers:
        layers.append(t[0])  # [H,Tq,Tk] for batch=1
    attn = torch.stack(layers, dim=0)  # [L,H,Tq,Tk]

    begin_pos_vis = MetadataStation.get_begin_pos('vis')
    vis_len = MetadataStation.get_vis_len()
    if begin_pos_vis is None or vis_len is None:
        raise RuntimeError("Missing visual token segmentation info.")
    attn_last_to_vis = attn[:, :, -1:, begin_pos_vis:begin_pos_vis + vis_len]
    return attn_last_to_vis


def _generate_collect(model, tokenizer, image_processor, input_ids, image_tensor, image_sizes, max_new_tokens=10, do_sample=False, num_beams=1):
    """Run generate to obtain output tokens, and try to collect attentions from the first generation step.

    Returns: (attn [L,H,1,V] or None, generated_text str)
    """
    gen = model.generate(
        inputs=input_ids,
        images=image_tensor.unsqueeze(0),
        image_sizes=image_sizes,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        return_dict_in_generate=True,
        output_attentions=True,
    )
    sequences = gen.sequences
    input_len = input_ids.shape[1]
    gen_ids = sequences[:, input_len:]
    generated_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

    attn_last_to_vis = None
    if hasattr(gen, 'attentions') and gen.attentions:
        step0 = gen.attentions[0]  # tuple length L, each [B,H,1,src]
        layers = [t[0] for t in step0]  # list of [H,1,src]
        attn = torch.stack(layers, dim=0)  # [L,H,1,src]
        begin_pos_vis = MetadataStation.get_begin_pos('vis')
        vis_len = MetadataStation.get_vis_len()
        if begin_pos_vis is None or vis_len is None:
            raise RuntimeError("Missing visual token segmentation info.")
        attn_last_to_vis = attn[:, :, :, begin_pos_vis:begin_pos_vis + vis_len]
    return attn_last_to_vis, generated_text


def collect_attention(cfg, image_file: str, query: str, save_dir: str, save_id: str) -> str:
    """Run one forward pass and save attention focused on image tokens.

    Saves a pickle with dict: {
      'attn': Tensor[L, H, 1, V],
      'meta': {image_file, query, image_size, model_name, vis_len, patch_size, num_layers, num_heads}
    }
    Returns the saved file path.
    """
    tokenizer, model, image_processor, _, model_name_str = load_model_from_cfg(cfg)

    # Prepare image
    image = load_image(image_file)
    image_size = image.size  # (W, H)
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    image_sizes = [image.size]

    # Prepare prompt
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in query:
        qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, query) if model.config.mm_use_im_start_end else re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, query)
    else:
        qs = (image_token_se + "\n" + query) if model.config.mm_use_im_start_end else (DEFAULT_IMAGE_TOKEN + "\n" + query)

    conv = conv_templates[cfg.model.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # Tokenize
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
        conv=conv,
    ).unsqueeze(0).to(model.device)

    # Collect attentions (and optional generated text)
    with torch.inference_mode():
        if getattr(cfg.model, 'use_generate', False):
            attn_last_to_vis, gen_text = _generate_collect(
                model, tokenizer, image_processor, input_ids, image_tensor, image_sizes,
                max_new_tokens=getattr(cfg.model, 'max_new_tokens', 10),
                do_sample=getattr(cfg.model, 'do_sample', False),
                num_beams=getattr(cfg.model, 'num_beams', 1),
            )
            if attn_last_to_vis is None:
                attn_last_to_vis = _forward_collect(
                    model, tokenizer, image_processor, input_ids, image_tensor, image_sizes
                )
        else:
            gen_text = None
            attn_last_to_vis = _forward_collect(
                model, tokenizer, image_processor, input_ids, image_tensor, image_sizes
            )

    P = int(np.sqrt(attn_last_to_vis.shape[-1]))
    meta = {
        "image_file": image_file,
        "query": query,
        "image_size": image_size,
        "model_name": model_name_str,
        "vis_len": int(attn_last_to_vis.shape[-1]),
        "patch_size": int(P),
        "num_layers": int(attn_last_to_vis.shape[0]),
        "num_heads": int(attn_last_to_vis.shape[1]),
    }
    if getattr(cfg.model, 'use_generate', False) and gen_text is not None:
        meta["generated_text"] = gen_text

    model_dir = _sanitize_name(cfg.model.name)
    out_dir = os.path.join(save_dir, model_dir)
    _ensure_dir(out_dir)

    save_path = os.path.join(out_dir, f"{save_id}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"attn": attn_last_to_vis.detach().cpu(), "meta": meta}, f)

    # Save a small side metadata for convenience
    with open(os.path.join(out_dir, f"{save_id}_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    return save_path
