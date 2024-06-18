# Fast-Compress

**This a WIP - do not use unless you are interested in contributing to the ongoing project.**

This repo extends [GPT-Fast](https://github.com/pytorch-labs/gpt-fast) by adding SOTA KV Cache compression methods.

When done, it *will* serve as an open-source, hackable toolkit to accelerate research onto memory efficient inference.

## Installation
[Download PyTorch nightly](https://pytorch.org/get-started/locally/)
```bash
pip install packaging ninja
MAX_JOBS=8 pip install flash-attn --no-build-isolation # Set MAX_JOBS to a lower value if you get OOM errors.
pip install -r requirements.txt
```

After logging in with `huggingface-cli login`, run

```bash
bash scripts/prepare_llama3.sh
```

This will create necessary model and tokenizer files for`Meta-Llama-3-8B-Instruct` within `./checkpoints`. It will also create a smaller model for debugging purposes only, called `Meta-Llama-3-8B-Instruct-4-Layers`. This model removes all layers except for the first 4. It's quicker to load but will generate nonsense, so only use for debugging.

## Usage

```
python generate.py --compile --cache_strategy full --prompt "short_prompt_long_output.txt"
```