import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch._dynamo.config
import torch._inductor.config
import torch.nn.functional as F
from datasets import load_dataset
from tqdm.auto import tqdm

from generate import _load_model, encode_tokens, model_forward
from model import Transformer
from tokenizer import get_tokenizer

torch._dynamo.config.automatic_dynamic_shapes = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.triton.cudagraphs = True
torch._dynamo.config.cache_size_limit = 100000

quality_multiple_choice_pattern = re.compile(r" *\([A-D]\) *")


def _normalize_answer(text):
    return " ".join(text.split()).strip()


def _drop_duplicates_in_input(untokenized_dataset):
    # from scrolls/evaluator/dataset_evaluator.py

    indices_to_keep = []
    id_to_idx = {}
    outputs = []
    for i, (id_, output) in enumerate(
        zip(untokenized_dataset["id"], untokenized_dataset["output"])
    ):
        if id_ in id_to_idx:
            outputs[id_to_idx[id_]].append(output)
            continue
        indices_to_keep.append(i)
        id_to_idx[id_] = len(outputs)
        outputs.append([output])
    untokenized_dataset = untokenized_dataset.select(indices_to_keep).flatten_indices()
    untokenized_dataset = untokenized_dataset.remove_columns("output")
    untokenized_dataset = untokenized_dataset.add_column("outputs", outputs)
    return untokenized_dataset


def _process_doc_prepended_question(doc):
    input = doc["input"]
    split = input.find("\n\n")
    return {
        "id": doc["id"],
        "pid": doc["pid"],
        "input": input,
        "outputs": doc["outputs"],
        "question": input[0:split],
        "text": input[split + 2 :],
    }


def process_doc(doc):
    doc = _process_doc_prepended_question(doc)

    split = doc["text"].find("\n\n", doc["text"].find("(D)"))
    choices_text = doc["text"][:split]

    doc["text"] = doc["text"][split:].strip()
    doc["choices"] = [
        _normalize_answer(choice)
        for choice in re.split(quality_multiple_choice_pattern, choices_text)[1:]
    ]
    doc["gold"] = doc["choices"].index(_normalize_answer(doc["outputs"][0]))
    return doc


def construct_requests(doc):
    ctx = f"{doc['text']}\n\nQuestion:{doc['question']}\n\nAnswer:"
    request_list = [
        {"context": ctx, "choice": " {}".format(choice), "idx": i}
        for i, choice in enumerate(doc["choices"])
    ]
    doc["requests"] = request_list
    return doc


def get_quality_dataset():
    """
    download and processes the quality dataset following the lm-evaluation-harness scrolls_quality task

    The processed dataset has the following train & validation splits with 2523 & 2086 examples respectively.
    fields to be used during evaluation:
    - question: the question prompt
    - text: the context
    - choices: list of choices (4 in total)
    - gold: index of the correct choice
    """
    quality_dataset = load_dataset("tau/scrolls", "quality")
    del quality_dataset["test"]  # drop test split -> no ground truths
    for split in quality_dataset:
        quality_dataset[split] = _drop_duplicates_in_input(quality_dataset[split])
    quality_dataset = quality_dataset.map(process_doc)
    quality_dataset = quality_dataset.map(construct_requests)
    return quality_dataset


def setup_cache_padded_seq_input_pos_max_seq_length_for_prefill(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    max_seq_length: Optional[int] = None,
):
    """
    Sets up model cache and does some bookkeeping calculations for prompt, input_pos and max_seq_length
    that are needed for prefill or model_forward

    Args:
        model (LLaMA): The model whose cache gets set up
        prompt (torch.Tensor): Tensor of shape (T) with indices of the prompt sequence.
        max_new_tokens (int): The desired maximum number of new tokens that can be generated.
        max_seq_length (Optional[int], optional): The maximum sequence length allowed.

    Returns:
        seq (torch.Tensor): prompt but padded with zeros to size max_seq_length
        input_pos (torch.Tensor): tensor of integers in increasing order
        max_seq_length (int): The maximum sequence length allowed, updated based on other numbers
    """
    T = prompt.size(0)
    T_new = T + max_new_tokens
    if max_seq_length is None:
        max_seq_length = min(T_new, model.config.block_size)

    device, dtype = prompt.device, prompt.dtype
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(T_new, dtype=dtype, device=device)
    empty[:T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    kwargs = {
        "cache_strategy": "full",
        "max_cache_length": [2048 for _ in range(len(model.layers))],
    }

    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length, **kwargs)

    return seq, input_pos, max_seq_length


@torch.no_grad()
def model_call(model, inps, max_length):
    inps = inps.squeeze(0)

    max_new_tokens = 1
    seq, input_pos, max_seq_length = (
        setup_cache_padded_seq_input_pos_max_seq_length_for_prefill(
            model,
            inps,
            max_new_tokens,
            max_length,
        )
    )

    x = seq.index_select(0, input_pos).view(1, -1)
    logits = model_forward(model, x, input_pos)
    return logits


def process_results(doc, lls):
    gold = doc["gold"]

    acc = 1.0 if np.argmax(lls) == gold else 0.0
    completion_len = np.array([float(len(i)) for i in doc["choices"]])
    acc_norm = 1.0 if np.argmax(lls / completion_len) == gold else 0.0
    return {"acc": acc, "acc_norm": acc_norm, "em": acc_norm * 100.0}


def tok_encode_quality(tokenizer, choice, context, device, max_seq_length=2048):
    encoded_choice = encode_tokens(tokenizer, choice, bos=False, device=device).tolist()
    encoded_context = encode_tokens(
        tokenizer, context, bos=False, device=device
    ).tolist()

    n_choice_tokens, n_context_tokens = len(encoded_choice), len(encoded_context)
    if n_choice_tokens + n_context_tokens + 1 > max_seq_length:
        n_context_tokens = max_seq_length - n_choice_tokens - 1

    tokens = [tokenizer.bos_id()] + encoded_context[-n_context_tokens:] + encoded_choice
    return torch.tensor(tokens, dtype=torch.int, device=device), encoded_choice


def main(checkpoint_path, limit=None, max_seq_length=2048):
    # load model & tokenizer ---
    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), str(tokenizer_path)

    device = "cuda"
    precision = torch.bfloat16

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision, False)

    torch.cuda.synchronize()
    print(f"Time to load model: {time.time() - t0:.02f} seconds.")
    model.eval()

    tokenizer = get_tokenizer(tokenizer_path, checkpoint_path)
    torch.manual_seed(1234)

    print("Downloading and processing scrolls/quality dataset")
    quality_ds = get_quality_dataset()
    ds = quality_ds["validation"]  # use validation split ---
    if limit is not None:
        print(f"Selecting first {limit} samples for evaluation")
        ds = ds.select(range(limit))
    print("Done")

    pbar = tqdm(range(len(ds)))
    results = []

    for doc in ds:
        requests = doc["requests"]

        lls = []
        for req in requests:
            choice = req["choice"]
            context = req["context"]

            inputs, choice_tokens = tok_encode_quality(
                tokenizer, choice, context, device, max_seq_length
            )

            logits = model_call(
                model, inputs, max_length=max_seq_length
            )  # (1, seq_len, vocab_size)

            n_choice_tokens = len(choice_tokens)
            choice_logits = logits[:, -n_choice_tokens:, :]
            multi_logits = F.log_softmax(
                choice_logits, dim=-1
            )  # (1, seq_len, vocab_size)
            choice_tokens = torch.tensor(
                choice_tokens, dtype=torch.int64, device=device
            )
            gathered_multi_logits = torch.gather(
                multi_logits, 2, choice_tokens.reshape(1, -1).unsqueeze(-1)
            ).squeeze(-1)
            ll = float(gathered_multi_logits.sum())
            lls.append(ll)
        r = process_results(doc, lls)
        results.append(r)
        pbar.update(1)
    pbar.close()

    # compute metrics ---
    acc = np.mean([r["acc"] for r in results])
    acc_norm = np.mean([r["acc_norm"] for r in results])
    em = np.mean([r["em"] for r in results])
    print(f"Accuracy: {acc:.02f}")
    print(f"Accuracy (normalized): {acc_norm:.02f}")
    print(f"EM: {em:.02f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("checkpoints/mistralai/Mistral-7B-Instruct-v0.1/model.pth"),
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--limit", type=int, default=16, help="number of samples to evalulate"
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="maximum length sequence to evaluate",
    )

    args = parser.parse_args()
    main(
        Path(args.checkpoint_path),
        args.limit,
        args.max_seq_length,
    )


# Example usage:
# python eval_quality.py --checkpoint_path checkpoints/mistralai/Mistral-7B-Instruct-v0.1/model.pth --limit 128 --max_seq_length 2048
# out:
# Accuracy: 0.24
# Accuracy (normalized): 0.33
# EM: 32.81


# TODO: add support for max length > 2048, for quality we will need ~8k max length
# TODO: add support for compile flag
# TODO: enable faster inference since context is the same for all choices
# TODO: double-check evaluation logic
# TODO: add support to run on multiple GPUs
