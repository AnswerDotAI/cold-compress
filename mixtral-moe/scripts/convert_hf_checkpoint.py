# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import glob
import json
import re
import sys
from pathlib import Path
from safetensors.torch import load
from typing import Optional

import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import ModelArgs


@torch.inference_mode()
def convert_hf_checkpoint(
    *,
    checkpoint_dir: Path = Path("checkpoints/databricks/dbrx-base"),
    model_name: Optional[str] = None,
) -> None:
    if model_name is None:
        model_name = checkpoint_dir.name

    config = ModelArgs.from_name(model_name)
    print(f"Model config {config.__dict__}")

    weight_map = {
        "transformer.wte.weight": "tok_embeddings.weight",
        "transformer.blocks.{}.norm_attn_norm.attn.Wqkv.weight": "layers.{}.attention.wqkv.weight",
        "transformer.blocks.{}.norm_attn_norm.attn.out_proj.weight": "layers.{}.attention.wo.weight",
        "transformer.blocks.{}.ffn.experts.mlp.w1": "layers.{}.block_sparse_moe.cond_ffn.w1",
        "transformer.blocks.{}.ffn.experts.mlp.w2": "layers.{}.block_sparse_moe.cond_ffn.w2",
        "transformer.blocks.{}.ffn.experts.mlp.v1": "layers.{}.block_sparse_moe.cond_ffn.w3",
        "transformer.blocks.{}.ffn.router.layer.weight": "layers.{}.block_sparse_moe.gate.weight",
        "transformer.blocks.{}.norm_attn_norm.norm_1.weight": "layers.{}.attention_norm.weight",
        "transformer.blocks.{}.norm_attn_norm.norm_2.weight": "layers.{}.ffn_norm.weight",
        "transformer.norm_f.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }

    st_files = glob.glob(str(checkpoint_dir / "*.safetensors"))

    merged_result = {}
    for file in sorted(st_files):
        with open(file, "rb") as f:
            data = f.read()
        state_dict = load(data)
        merged_result.update(state_dict)
    final_result = {}
    for key, value in merged_result.items():
        print("original key = ", key)
        if "blocks" in key:
            abstract_key = re.sub(r'.(\d+).', '.{}.', key, count=1)
            layer_num = re.search(r'\d+', key).group(0)
            new_key = weight_map[abstract_key]
            if new_key is None:
                continue
            new_key = new_key.format(layer_num)
            print("1 new_key = ", new_key)
        else:
            new_key = weight_map[key]
            print("2 new_key = ", new_key)

        final_result[new_key] = value

    for key in tuple(final_result.keys()):
        if "w1" in key or "w3" in key:
            final_result[key] = final_result[key].reshape(config.num_experts, config.intermediate_size, config.dim).contiguous()
        elif "w2" in key:
            final_result[key] = final_result[key].reshape(config.num_experts, config.intermediate_size, config.dim).permute(0, 2, 1).contiguous()
        elif "gate" in key:
            final_result[key] = final_result[key].contiguous()

    print(f"Saving checkpoint to {checkpoint_dir / 'model.pth'}")
    torch.save(final_result, checkpoint_dir / "model.pth")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert HuggingFace checkpoint.')
    parser.add_argument('--checkpoint_dir', type=Path, default=Path("checkpoints/meta-llama/llama-2-7b-chat-hf"))
    parser.add_argument('--model_name', type=str, default=None)

    args = parser.parse_args()
    convert_hf_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
    )
