import argparse
import os

from claudette import models as anthropic_models, Chat

from datasets import Dataset
import pandas as pd
import torch
from data_utils import BENCHMARKS
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

assert (
    "ANTHROPIC_API_KEY" in os.environ
), "Please set the ANTHROPIC_API_KEY environment variable."


PROMPT_TEMPLATES = {
    "triviaqa": "Compress the information in the retrieved documents into a 1-3 sentences "
    "such that it includes only information relevant to answering the question: %s\n\nRetrieved Documents:\n%s",
    "dolomites": "Compress the information in the instructions into  a 1-3 sentences "
    "such that it includes only information relevant to completing the task: %s\n\nInstructions:\n%s"
}

SUMMARY_PREFILL = {"triviaqa": "Compressed Documents: ",
                   "dolomites": "Compressed Instructions: ",}

SCORE_PREFILL = {"triviaqa": "The answer is",
                 "dolomites": "The completed task is",}

LONGCONTEXT_DATASETS = ["dolomites"] # no RAG will be the same as original

# We will evaluate each summary based on if it improves downstream performance:
# p(answer|question, summarized context) minus either p(answer|original context) or p(answer|question)
SCORER_MODEL = (
    "microsoft/Phi-3-mini-128k-instruct"  # "meta-llama/Meta-Llama-3-8B-Instruct"
)


def compute_likelihoods(
    args, ctx, q, answers, instruction, tokenizer, scorer, max_ctx_len=None, batch_size=8
):
    # We need enough tokens for the instruction, question, and answer (subtract max context by 1024 to be safe)
    max_ctx_len = max_ctx_len or tokenizer.model_max_length - 1024
    ctx_tokens = len(tokenizer.encode(ctx))
    if ctx_tokens > max_ctx_len:
        ctx_words = ctx.split(" ")
        keep_num_words = round(len(ctx_words) * max_ctx_len / ctx_tokens)
        ctx = " ".join(ctx_words[:keep_num_words])

    chat = [
        {"role": "user", "content": f"{ctx}\n\n{instruction}\n\n{q}"},
    ]

    prompt = (
        tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        + SCORE_PREFILL[args.dataset]
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    n = len(prompt_ids)

    ans_ids = [
        tokenizer.encode(" " + ans.strip(), add_special_tokens=False) for ans in answers
    ]
    input_ids = [prompt_ids + ans_id for ans_id in ans_ids]
    max_len = max(map(len, input_ids))

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pad the input_ids up to max_len
    assert tokenizer.padding_side in {"right", "left"}
    input_ids = torch.LongTensor(
        [
            ids + [tokenizer.pad_token_id] * (max_len - len(ids))
            if tokenizer.padding_side == "right"
            else [tokenizer.pad_token_id] * (max_len - len(ids)) + ids
            for ids in input_ids
        ]
    )
    input_ids = input_ids.to(scorer.device)

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")

    labels = input_ids.clone()
    # No Loss on the prompt
    labels[:, :n] = -100
    is_pad = input_ids == tokenizer.pad_token_id
    labels[is_pad] = -100
    shift_labels = labels[..., 1:].contiguous()

    j = 0
    scores = []
    for batch_idx in range(0, len(input_ids), batch_size):
        with torch.no_grad():
            logits = scorer(input_ids[batch_idx : batch_idx + batch_size]).logits
            shift_logits = logits[..., :-1, :].contiguous()
            for i in range(len(shift_logits)):
                _shift_logits = shift_logits[i].view(-1, scorer.config.vocab_size)
                _shift_labels = shift_labels[j].view(-1)
                ll = (-loss_fct(_shift_logits, _shift_labels)).item()
                scores.append(ll)
                j += 1

    return {"max": max(scores), "mean": sum(scores) / len(scores)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name", type=str, default="haiku", choices=["haiku", "sonnet", "opus"]
    )
    parser.add_argument("--dataset", type=str, default="triviaqa")
    parser.add_argument(
        "--min_toks",
        type=int,
        default=50,
        help="If context has < min_toks, there is no need to summarize it. The summary is itself.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Max batch size for the scorer."
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
        default=".",
        help="The directory to save the summarized dataset.",
    )

    args = parser.parse_args()

    model = [m for m in anthropic_models if args.name in str(m)][0]

    dataset = BENCHMARKS[args.dataset]()

    train = dataset.get_train()

    scorer = (
        AutoModelForCausalLM.from_pretrained(
            SCORER_MODEL,
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        .eval()
        .to("cuda")
    )

    tokenizer = AutoTokenizer.from_pretrained(SCORER_MODEL)

    stats = []
    out_data = []

    for ex in tqdm(train):
        # Claudette is stateful so you need to re-instantiate each time
        chat = Chat(model, sp="""You are a helpful and concise assistant.""")

        q, ctx = dataset.question(ex), dataset.context(ex)
        ctx_toks = len(ctx.split(" "))

        if ctx_toks < args.min_toks and args.dataset not in LONGCONTEXT_DATASETS:
            summary = ctx            
        else:
            prompt = PROMPT_TEMPLATES[args.dataset] % (q, ctx)

            summary = chat(prompt, prefill=SUMMARY_PREFILL[args.dataset]).content[0].text

            assert summary.startswith(SUMMARY_PREFILL[args.dataset])
            summary = summary[len(SUMMARY_PREFILL[args.dataset]) :].strip()

            answers = dataset.answer(ex)

            if type(answers) == str:
                answers = [answers]

            original_scores = compute_likelihoods(
                args,
                ctx,
                q,
                answers,
                dataset.instruction(),
                tokenizer,
                scorer,
                max_ctx_len=args.max_ctx_len,
                batch_size=args.batch_size,
            )
            compressed_scores = compute_likelihoods(
                args,
                summary,
                q,
                answers,
                dataset.instruction(),
                tokenizer,
                scorer,
                max_ctx_len=args.max_ctx_len,
                batch_size=args.batch_size,
            )

            if args.dataset not in LONGCONTEXT_DATASETS:
                no_rag_scores = compute_likelihoods(
                    args,
                    "",
                    q,
                    answers,
                    dataset.instruction(),
                    tokenizer,
                    scorer,
                    max_ctx_len=args.max_ctx_len,
                    batch_size=args.batch_size,
                )

            else: no_rag_scores = original_scores

            stats.append(
                {
                    "original_mean": original_scores["mean"],
                    "original_max": original_scores["max"],
                    "compressed_mean": compressed_scores["mean"],
                    "compressed_max": compressed_scores["max"],
                    "no_rag_mean": no_rag_scores["mean"],
                    "no_rag_max": no_rag_scores["max"],
                }
            )

            print("Running statistics...")
            print(pd.DataFrame(stats).mean())

            out_row = ex.copy()
            out_row.update({"summary_prompt": prompt, "summary": summary})
            out_data.append(out_row)

    dataset = Dataset.from_list(out_data)
    out_hf_path = os.path.join(args.out_dir, f"{args.dataset}_summarized")
    print(f"Saving data with downstream scores to {out_hf_path}")
    dataset.save_to_disk(out_hf_path)
