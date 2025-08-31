# Deprecated LLaMA modeling fork

The original local fork of LLaMA modeling used by LLaVA has been deprecated in the overhaul pipeline.
We now rely on Hugging Face Transformers' upstream `LlamaModel`/`LlamaForCausalLM` implementations and
collect attentions via the standard `output_attentions=True` API.

This directory exists to record that the local fork has been retired. Source remains in
`_overhaul/llava/model/llm_modeling/` but should not be imported by the overhaul.

