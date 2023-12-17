# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import re
import sys
from pathlib import Path
from typing import Optional

import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import ModelArgs
import glob


@torch.inference_mode()
def convert_hf_checkpoint(
    *,
    checkpoint_dir: Path = Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf"),
    model_name: Optional[str] = None,
) -> None:
    if model_name is None:
        model_name = checkpoint_dir.name

    config = ModelArgs.from_name(model_name)
    print(f"Model config {config.__dict__}")

    weight_map = {
        "tok_embeddings.weight": "tok_embeddings.weight",
        "layers.{}.attention.wq.weight": "layers.{}.attention.wq.weight",
        "layers.{}.attention.wk.weight": "layers.{}.attention.wk.weight",
        "layers.{}.attention.wv.weight": "layers.{}.attention.wv.weight",
        "layers.{}.attention.wo.weight": "layers.{}.attention.wo.weight",
        "layers.{}.block_sparse_moe.w1": "layers.{}.block_sparse_moe.cond_ffn.w1",
        "layers.{}.block_sparse_moe.w2": "layers.{}.block_sparse_moe.cond_ffn.w2",
        "layers.{}.block_sparse_moe.w3": "layers.{}.block_sparse_moe.cond_ffn.w3",
        "layers.{}.block_sparse_moe.gate.weight": "layers.{}.block_sparse_moe.gate.weight",
        "layers.{}.attention_norm.weight": "layers.{}.attention_norm.weight",
        "layers.{}.ffn_norm.weight": "layers.{}.ffn_norm.weight",
        "norm.weight": "norm.weight",
        "output.weight": "output.weight",
    }
    pt_files = glob.glob(str(checkpoint_dir / "*.pt"))

    def permute(w, n_head):
        dim = config.dim
        return (
            w.view(n_head, 2, config.head_dim // 2, dim)
            .transpose(1, 2)
            .reshape(config.head_dim * n_head, dim)
        )

    merged_result = {}
    for file in sorted(pt_files):
        state_dict = torch.load(str(file), map_location="cpu", mmap=True, weights_only=True)
        merged_result.update(state_dict)
    final_result = {}
    for key, value in merged_result.items():
        if "layers" in key:
            abstract_key = re.sub(r'.(\d+).', '.{}.', key)
            layer_num = re.search(r'\d+', key).group(0)
            new_key = weight_map[abstract_key]
            if new_key is None:
                continue
            new_key = new_key.format(layer_num)
        else:
            new_key = weight_map[key]

        final_result[new_key] = value

    for key in tuple(final_result.keys()):
        if "wq" in key:
            q = final_result[key]
            k = final_result[key.replace("wq", "wk")]
            v = final_result[key.replace("wq", "wv")]
            # q = permute(q, config.n_head)
            # k = permute(k, config.n_local_heads)
            final_result[key.replace("wq", "wqkv")] = torch.cat([q, k, v])
            del final_result[key]
            del final_result[key.replace("wq", "wk")]
            del final_result[key.replace("wq", "wv")]
        if "w1" in key or "w2" in key or "w3" in key:
            final_result[key] = final_result[key].reshape(config.num_experts, config.intermediate_size, config.dim).contiguous()
        if "gate" in key:
            final_result[key] = final_result[key].contiguous()
        # if "w1" in key:
        #     final_result[key] = final_result[key].reshape(config.num_experts, config.intermediate_size, config.dim)
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
