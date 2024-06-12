import argparse
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from task import AutoTask
from evaluate import load
import pandas as pd
from scipy.stats import spearmanr
import numpy as np
np.random.seed(1992)
from tqdm import tqdm

try:
    import nltk

    nltk.sent_tokenize("test")
except:
    nltk.download("punkt")


ROUGE = load("rouge", keep_in_memory=True)
_DEFAULT_SCORE_PREFILL = "The answer is"
SCORE_PREFILL = {"dolomites": "The completed task is"}


def prepare_inputs(row, task, ctxs, tokenizer, max_ctx_len=None):
    # We need enough tokens for the instruction, question, and answer (subtract max context by 1024 to be safe)
    max_ctx_len = max_ctx_len or tokenizer.model_max_length - 1024

    inputs = []
    assert row["context"] in row["prompt"]
    for i, ctx in enumerate(ctxs):
        ctx_token = len(tokenizer.encode(ctx))
        if ctx_token > max_ctx_len:
            ctx_words = ctx.split(" ")
            keep_num_words = round(len(ctx_words) * max_ctx_len / ctx_token)
            ctxs[i] = " ".join(ctx_words[:keep_num_words])
        prompt = task.form_prompt(
            **{
                "context": ctx,
                "question": row["question"],
            }
        )

        prompt = [
            {"role": "user", "content": prompt}
        ]

        if "qwen" in args.model_name.lower():
            prompt.insert(0, {"role": "system", "content": "You are a helpful assistant."})

        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        prompt += SCORE_PREFILL.get(task.name, _DEFAULT_SCORE_PREFILL)
        inputs.append(tokenizer.encode(prompt, add_special_tokens=False))
    return inputs


def compute_extractive_labels(args, task, scorer, row):
    sents = nltk.sent_tokenize(row["context"] or row["question"])
    labels = row["labels"]

    input_ids = prepare_inputs(
        row, task, sents, tokenizer, max_ctx_len=args.max_ctx_len
    )
    input_lens = list(map(len, input_ids))
    rouge = ROUGE.compute(
        predictions=sents,
        rouge_types=["rouge1", "rouge2", "rougeL"],
        references=[labels for _ in range(len(sents))],
        use_aggregator=False,
    )
    outputs = {
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
    }

    # TODO: We are taking the first label only -- not ideal if multiple diverse good references
    label = labels[0]
    label_ids = tokenizer.encode(" " + label.strip(), add_special_tokens=False)

    input_ids = [input + label_ids for input in input_ids]

    max_len = max(map(len, input_ids))

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pad the input_ids up to max_len
    assert tokenizer.padding_side in {"right", "left"}
    input_ids_pad = torch.LongTensor(
        [
            ids + [tokenizer.pad_token_id] * (max_len - len(ids))
            if tokenizer.padding_side == "right"
            else [tokenizer.pad_token_id] * (max_len - len(ids)) + ids
            for ids in input_ids
        ]
    )

    # check if any infs or nans in input_ids
    input_ids_pad = input_ids_pad.to(scorer.device)
    assert not torch.isnan(input_ids_pad).any()

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")

    labels = input_ids_pad.clone()
    # No Loss on the prompt
    for i, l in enumerate(input_lens):
        labels[i, :l] = -100
    is_pad = input_ids_pad == tokenizer.pad_token_id
    labels[is_pad] = -100
    shift_labels = labels[..., 1:].contiguous()

    j = 0
    scores = []
    for batch_idx in range(0, len(input_ids_pad), args.batch_size):
        with torch.no_grad():
            logits = scorer(
                input_ids_pad[batch_idx : batch_idx + args.batch_size]
            ).logits
            assert not torch.isnan(logits).any()
            shift_logits = logits[..., :-1, :].contiguous()
            for i in range(len(shift_logits)):
                _shift_logits = shift_logits[i].view(-1, scorer.config.vocab_size)
                _shift_labels = shift_labels[j].view(-1)
                assert not torch.isnan(_shift_labels).any()
                assert not torch.isnan(_shift_logits).any()
                ll = (-loss_fct(_shift_logits, _shift_labels)).item()
                scores.append(ll)
                j += 1

    outputs["sents"] = sents
    outputs["model_scores"] = scores

    print("Question:")
    print(row["question"])

    print("Top sentence by model score:")
    print(sents[np.argmax(scores)])
    print("\nTop sentence by ROUGE-1:")
    print(sents[np.argmax(outputs["rouge1"])])

    corels = {
        r: spearmanr(scores, outputs[r])[0] for r in ["rouge1", "rouge2", "rougeL"]
    }
    return outputs, corels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/Phi-3-mini-128k-instruct",  # "Qwen/Qwen2-7B-Instruct",
        help="The model to use as the extractor.",
    )

    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="The maximum number of examples to process for each split.",
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["squality"],
        help="The dataset to use.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Max batch size for the scorer.",
    )

    parser.add_argument(
        "--max_ctx_len",
        type=int,
        default=None,
        help="An optional override to max context window of model.",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        # Default dir is the parent directory of this file in a "data" folder
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
        help="The output directory.",
    )

    args = parser.parse_args()

    # TODO: Actually account for multiple datasets
    task = AutoTask.from_name(args.tasks[0])

    out_hf_path = os.path.join(args.out_dir, f"{task.name}_extracted")
    os.makedirs(out_hf_path, exist_ok=True)

    print(f"Initializing scorer from {args.model_name}")
    scorer = (
        AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        .eval()
        .to("cuda")
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    stats = []
    out_data = []

    dataset = {
        "train": task.get_train(),
        "validation": task.get_validation(),
        "test": task.get_test(),
    }

    for split, data in dataset.items():
        n = len(data)
        if args.max_examples is not None and n > args.max_examples:
            print(f"Subsampling {args.max_examples} from {n}...")
            idxs = np.arange(n)
            np.random.shuffle(idxs)
            data = data.select(idxs[: args.max_examples])
            n = args.max_examples
        print(f"Processing {n} examples from {split}...")
        corels = []
        for row in tqdm(data):
            new_data, _corels = compute_extractive_labels(args, task, scorer, row)

            corels.append(_corels)
            print("Avg Rank Correlations of ROUGE to model scores:\n")
            print(pd.DataFrame(corels).mean())

        dataset[split] = data.map(lambda _, idx: new_data[idx], with_indices=True)

    print(f"Saving to {out_hf_path}")
    dataset.save_to_disk(out_hf_path)
